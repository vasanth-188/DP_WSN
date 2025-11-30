#!/usr/bin/env python3
"""
wsn_bridge.py
UDP bridge for WSN DQN simulation.
Receives JSON state from ns-3 simulation, processes it, and sends back actions.
Includes real-time visualization of energy, packet counts, and node activity.
"""

import socket
import json
import sys
import time
from typing import Dict, Any, List
import threading
from collections import defaultdict

# Try to import matplotlib, but don't fail if it's not available
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib not installed. Plotting disabled.")
    print("[INFO] Install with: pip3 install matplotlib")
    print("[INFO] Run with --no-plot to suppress this warning")

# Configuration
NS3_SEND_ADDR = "127.0.0.1"
NS3_SEND_PORT = 5000
NS3_RECV_PORT = 5001

class WSNBridge:
    def __init__(self, enable_plot=True):
        """Initialize the WSN bridge with UDP sockets and optional live plotting."""
        # Socket to send actions to ns-3
        self.tx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tx_socket.settimeout(5.0)
        
        # Socket to receive state from ns-3
        self.rx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx_socket.bind(("127.0.0.1", NS3_RECV_PORT))
        self.rx_socket.settimeout(2.0)
        
        self.running = True
        self.episode_count = 0
        self.step_count = 0
        
        # Data collection for plotting
        self.history = {
            "epochs": [],
            "node_energy": defaultdict(list),
            "node_tx": defaultdict(list),
            "node_rx": defaultdict(list),
            "awake_count": [],
            "total_tx": [],
            "total_rx": [],
        }
        self.enable_plot = enable_plot and MATPLOTLIB_AVAILABLE
        
        print(f"[WSNBridge] Initialized.")
        print(f"[WSNBridge] TX socket ready to send to {NS3_SEND_ADDR}:{NS3_SEND_PORT}")
        print(f"[WSNBridge] RX socket listening on 127.0.0.1:{NS3_RECV_PORT}")
        if enable_plot:
            if MATPLOTLIB_AVAILABLE:
                print(f"[WSNBridge] Live plotting enabled (matplotlib)")
                self._setup_plot()
            else:
                print(f"[WSNBridge] Plotting disabled (matplotlib not installed)")

    def simple_policy(self, state: Dict[str, Any]) -> List[int]:
        """
        Simple greedy policy: wake the 5 nodes with lowest energy.
        Returns a list of node IDs to wake up.
        
        Args:
            state: Dict containing 'nodes' list with per-node state
            
        Returns:
            List of node IDs to wake (1-indexed)
        """
        nodes = state.get("nodes", [])
        
        # Sort by energy (ascending) and take top 5 to wake
        sorted_nodes = sorted(nodes, key=lambda x: x.get("energy", 0))
        wake_list = [node["id"] for node in sorted_nodes[:5]]
        
        return wake_list

    def _setup_plot(self):
        """Setup matplotlib figure with subplots for real-time visualization."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("WSN DQN Real-Time Monitoring")
        
        # Subplot 1: Energy over time
        self.ax_energy = self.axes[0, 0]
        self.ax_energy.set_title("Node Energy over Time")
        self.ax_energy.set_xlabel("Epoch (s)")
        self.ax_energy.set_ylabel("Energy (J)")
        self.ax_energy.grid(True, alpha=0.3)
        
        # Subplot 2: TX/RX packets over time
        self.ax_packets = self.axes[0, 1]
        self.ax_packets.set_title("Total TX/RX Packets over Time")
        self.ax_packets.set_xlabel("Epoch (s)")
        self.ax_packets.set_ylabel("Packet Count")
        self.ax_packets.grid(True, alpha=0.3)
        
        # Subplot 3: Awake nodes count
        self.ax_awake = self.axes[1, 0]
        self.ax_awake.set_title("Awake Nodes Count")
        self.ax_awake.set_xlabel("Epoch (s)")
        self.ax_awake.set_ylabel("Count")
        self.ax_awake.grid(True, alpha=0.3)
        
        # Subplot 4: Per-node energy distribution (latest)
        self.ax_dist = self.axes[1, 1]
        self.ax_dist.set_title("Latest Energy Distribution")
        self.ax_dist.set_xlabel("Node ID")
        self.ax_dist.set_ylabel("Energy (J)")
        self.ax_dist.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.ion()  # Enable interactive mode

    def _update_plot(self):
        """Update all plot subplots with latest data."""
        if not self.enable_plot or not MATPLOTLIB_AVAILABLE or not self.history["epochs"]:
            return
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        epochs = self.history["epochs"]
        
        # Plot 1: Energy trends for selected nodes
        self.ax_energy.set_title("Node Energy over Time (Sample Nodes)")
        self.ax_energy.set_xlabel("Epoch (s)")
        self.ax_energy.set_ylabel("Energy (J)")
        for node_id in [0, 1, 5, 10, 24]:  # Sample nodes
            if node_id in self.history["node_energy"]:
                self.ax_energy.plot(epochs, self.history["node_energy"][node_id], 
                                   label=f"Node {node_id}", alpha=0.7)
        self.ax_energy.legend(loc="best")
        self.ax_energy.grid(True, alpha=0.3)
        
        # Plot 2: Total TX/RX packets
        self.ax_packets.set_title("Total TX/RX Packets over Time")
        self.ax_packets.set_xlabel("Epoch (s)")
        self.ax_packets.set_ylabel("Packet Count")
        self.ax_packets.plot(epochs, self.history["total_tx"], label="Total TX", marker='o', alpha=0.7)
        self.ax_packets.plot(epochs, self.history["total_rx"], label="Total RX", marker='s', alpha=0.7)
        self.ax_packets.legend(loc="best")
        self.ax_packets.grid(True, alpha=0.3)
        
        # Plot 3: Awake nodes count
        self.ax_awake.set_title("Awake Nodes Count")
        self.ax_awake.set_xlabel("Epoch (s)")
        self.ax_awake.set_ylabel("Count")
        self.ax_awake.bar(range(len(epochs)), self.history["awake_count"], alpha=0.7, color='green')
        self.ax_awake.grid(True, alpha=0.3)
        
        # Plot 4: Latest energy distribution
        if self.history["epochs"]:
            latest_idx = -1
            latest_energy = [self.history["node_energy"][i][latest_idx] if i in self.history["node_energy"] 
                           and len(self.history["node_energy"][i]) > 0 else 0 for i in range(25)]
            self.ax_dist.set_title("Latest Energy Distribution")
            self.ax_dist.set_xlabel("Node ID")
            self.ax_dist.set_ylabel("Energy (J)")
            colors = ['green' if e > 50 else 'orange' if e > 20 else 'red' for e in latest_energy]
            self.ax_dist.bar(range(25), latest_energy, color=colors, alpha=0.7)
            self.ax_dist.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)  # Small pause to update display

    def run(self):
        """Main event loop: receive state, compute action, send action."""
        print("\n[WSNBridge] Waiting for state from ns-3 simulation...\n")
        
        try:
            while self.running:
                try:
                    # Receive state from ns-3
                    data, addr = self.rx_socket.recvfrom(4096)
                    state_json = json.loads(data.decode("utf-8"))
                    
                    epoch = state_json.get("epoch", 0)
                    num_nodes = state_json.get("num_nodes", 0)
                    nodes = state_json.get("nodes", [])
                    
                    # Record history for plotting
                    self.history["epochs"].append(epoch)
                    awake_count = 0
                    total_tx = 0
                    total_rx = 0
                    
                    for node_data in nodes:
                        node_id = node_data.get("id", 0)
                        energy = node_data.get("energy", 0)
                        tx = node_data.get("tx", 0)
                        rx = node_data.get("rx", 0)
                        awake = node_data.get("awake", False)
                        
                        self.history["node_energy"][node_id].append(energy)
                        self.history["node_tx"][node_id].append(tx)
                        self.history["node_rx"][node_id].append(rx)
                        
                        if awake:
                            awake_count += 1
                        total_tx += tx
                        total_rx += rx
                    
                    self.history["awake_count"].append(awake_count)
                    self.history["total_tx"].append(total_tx)
                    self.history["total_rx"].append(total_rx)
                    
                    print(f"[Epoch {epoch:.2f}] Received state for {num_nodes} nodes | Awake: {awake_count} | TX: {total_tx} | RX: {total_rx}")
                    
                    # Compute action using simple policy
                    wake_list = self.simple_policy(state_json)
                    
                    # Send action back to ns-3
                    action = {"wake_list": wake_list}
                    action_json = json.dumps(action)
                    self.tx_socket.sendto(action_json.encode("utf-8"), (NS3_SEND_ADDR, NS3_SEND_PORT))
                    
                    print(f"           Sent action: wake_list={wake_list}")
                    self.step_count += 1
                    
                    # Update plot every 5 steps to avoid slowdown
                    if self.step_count % 5 == 0:
                        self._update_plot()
                    
                except socket.timeout:
                    # Timeout is expected periodically
                    continue
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse JSON: {e}")
                    continue
                except Exception as e:
                    print(f"[ERROR] Unexpected error: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\n[WSNBridge] Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Close sockets, save plot, and cleanup."""
        self.rx_socket.close()
        self.tx_socket.close()
        print(f"[WSNBridge] Processed {self.step_count} epochs.")
        
        # Save final plot
        if self.enable_plot:
            self._update_plot()
            try:
                plt.savefig("wsn_monitor.png", dpi=150, bbox_inches="tight")
                print(f"[WSNBridge] Plot saved to wsn_monitor.png")
            except Exception as e:
                print(f"[WARNING] Could not save plot: {e}")
            
            try:
                plt.show()
            except:
                pass
        
        # Save history to JSON for post-processing
        history_data = {
            "epochs": self.history["epochs"],
            "node_energy": {str(k): v for k, v in self.history["node_energy"].items()},
            "node_tx": {str(k): v for k, v in self.history["node_tx"].items()},
            "node_rx": {str(k): v for k, v in self.history["node_rx"].items()},
            "awake_count": self.history["awake_count"],
            "total_tx": self.history["total_tx"],
            "total_rx": self.history["total_rx"],
        }
        try:
            with open("wsn_history.json", "w") as f:
                json.dump(history_data, f, indent=2)
            print(f"[WSNBridge] History saved to wsn_history.json")
        except Exception as e:
            print(f"[WARNING] Could not save history: {e}")
        
        print(f"[WSNBridge] Goodbye!")

if __name__ == "__main__":
    # Check for --no-plot flag to disable visualization
    enable_plot = "--no-plot" not in sys.argv
    
    bridge = WSNBridge(enable_plot=enable_plot)
    bridge.run()
