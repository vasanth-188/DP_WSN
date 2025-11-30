#!/usr/bin/env python3
"""
wsn_dqn_agent.py
DQN Agent for WSN Wake/Sleep Scheduling

The agent learns to select which nodes should be awake each epoch to:
- Maximize packet delivery to sink
- Minimize energy consumption
- Balance network lifetime vs throughput

Reward = α × packets_delivered - β × awake_nodes - γ × dead_nodes

Policies available:
1. DQN (learned)
2. Always-On (baseline)
3. Round-Robin (heuristic)
4. Random (baseline)
5. Greedy-Energy (wake lowest energy nodes)
"""

import socket
import json
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Any
import time

# Try to import PyTorch for DQN
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    print("[WARNING] PyTorch not installed. DQN disabled, using heuristics only.")
    print("[INFO] Install with: pip3 install torch")

# Configuration
NS3_ADDR = "127.0.0.1"
NS3_RECV_PORT = 5000  # We receive state from ns-3 here
NS3_SEND_PORT = 5001  # We send actions to ns-3 here

# Reward weights
ALPHA = 1.0    # Reward per packet delivered
BETA = 0.1    # Penalty per awake node
GAMMA = 5.0    # Penalty per dead node (energy <= 0)

# DQN Hyperparameters
LEARNING_RATE = 0.001
GAMMA_DISCOUNT = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 10


# DQN Classes - only defined if PyTorch is available
if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network for node selection."""
        
        def __init__(self, state_size: int, num_nodes: int):
            super(DQNNetwork, self).__init__()
            self.num_nodes = num_nodes
            
            # Network outputs Q-value for each node being awake/asleep
            # Action space: 2^num_nodes is too large, so we use per-node Q-values
            self.fc1 = nn.Linear(state_size, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, num_nodes)  # Q-value for waking each node
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            return torch.sigmoid(self.fc4(x))  # 0-1 probability of waking each node


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class WSNAgent:
    """WSN DQN Agent with multiple policy options."""
    
    def __init__(self, num_nodes: int = 25, policy: str = "dqn"):
        self.num_nodes = num_nodes
        self.policy = policy.lower()
        self.episode = 0
        self.step = 0
        self.epsilon = EPSILON_START
        
        # State: [energy_1, ..., energy_N, awake_1, ..., awake_N, tx_1, ..., tx_N]
        self.state_size = num_nodes * 3  # energy, awake, tx_rate per node
        
        # Track metrics
        self.total_reward = 0
        self.rewards_history = []
        self.packets_history = []
        self.energy_history = []
        self.awake_history = []
        
        # Previous state for reward calculation
        self.prev_state = None
        self.prev_action = None
        self.prev_rx = 0
        
        # Round-robin state
        self.rr_offset = 0
        
        # Initialize DQN if available and requested
        self.dqn_enabled = TORCH_AVAILABLE and policy == "dqn"
        if self.dqn_enabled:
            self._init_dqn()
        
        print(f"[WSNAgent] Initialized with policy: {self.policy}")
        print(f"[WSNAgent] Num nodes: {num_nodes}, State size: {self.state_size}")
        if self.dqn_enabled:
            print(f"[WSNAgent] DQN enabled with ε={self.epsilon:.2f}")
    
    def _init_dqn(self):
        """Initialize DQN networks and optimizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQNNetwork(self.state_size, self.num_nodes - 1).to(self.device)
        self.target_net = DQNNetwork(self.state_size, self.num_nodes - 1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        print(f"[WSNAgent] DQN initialized on {self.device}")
    
    def state_to_vector(self, state: Dict) -> np.ndarray:
        """Convert JSON state to numpy vector."""
        nodes = state.get("nodes", [])
        
        # Normalize features
        energies = np.array([n.get("energy", 0) / 100.0 for n in nodes])
        awakes = np.array([1.0 if n.get("awake", False) else 0.0 for n in nodes])
        tx_rates = np.array([min(n.get("tx", 0) / 100.0, 1.0) for n in nodes])
        
        return np.concatenate([energies, awakes, tx_rates]).astype(np.float32)
    
    def calculate_reward(self, state: Dict) -> float:
        """Calculate reward based on current state."""
        nodes = state.get("nodes", [])
        
        # Packets delivered (sink's RX count increase)
        sink_rx = nodes[0].get("rx", 0) if nodes else 0
        packets_delivered = sink_rx - self.prev_rx
        self.prev_rx = sink_rx
        
        # Count awake nodes (excluding sink)
        awake_count = sum(1 for n in nodes[1:] if n.get("awake", False))
        
        # Count dead nodes (energy <= 0)
        dead_count = sum(1 for n in nodes[1:] if n.get("energy", 0) <= 0)
        
        # Reward calculation
        reward = (ALPHA * packets_delivered 
                  - BETA * awake_count 
                  - GAMMA * dead_count)
        
        return reward, packets_delivered, awake_count, dead_count
    
    def select_action(self, state: Dict) -> List[int]:
        """Select which nodes to wake based on policy."""
        if self.policy == "always_on":
            return self._policy_always_on()
        elif self.policy == "round_robin":
            return self._policy_round_robin()
        elif self.policy == "random":
            return self._policy_random(state)
        elif self.policy == "greedy_energy":
            return self._policy_greedy_energy(state)
        elif self.policy == "dqn" and self.dqn_enabled:
            return self._policy_dqn(state)
        else:
            # Default to always-on if DQN not available
            return self._policy_always_on()
    
    def _policy_always_on(self) -> List[int]:
        """Baseline: all nodes always awake."""
        return [1] * self.num_nodes
    
    def _policy_round_robin(self) -> List[int]:
        """Heuristic: rotate which 50% of nodes are awake."""
        wake_list = [1]  # Sink always on
        
        # Wake half the nodes in a rotating pattern
        num_sensor_nodes = self.num_nodes - 1
        nodes_to_wake = num_sensor_nodes // 2
        
        for i in range(1, self.num_nodes):
            # Node i is awake if it falls within the current window
            idx = (i - 1)  # 0-indexed for sensors
            window_start = self.rr_offset % num_sensor_nodes
            window_end = (window_start + nodes_to_wake) % num_sensor_nodes
            
            if window_start <= window_end:
                awake = window_start <= idx < window_end
            else:
                awake = idx >= window_start or idx < window_end
            
            wake_list.append(1 if awake else 0)
        
        # Advance round-robin offset
        self.rr_offset = (self.rr_offset + 3) % num_sensor_nodes
        
        return wake_list
    
    def _policy_random(self, state: Dict) -> List[int]:
        """Baseline: randomly wake 50% of nodes."""
        wake_list = [1]  # Sink always on
        for i in range(1, self.num_nodes):
            wake_list.append(1 if random.random() > 0.5 else 0)
        return wake_list
    
    def _policy_greedy_energy(self, state: Dict) -> List[int]:
        """Heuristic: wake nodes with highest energy."""
        nodes = state.get("nodes", [])
        wake_list = [1]  # Sink always on
        
        # Sort sensor nodes by energy (descending)
        sensor_nodes = [(i, n.get("energy", 0)) for i, n in enumerate(nodes) if i > 0]
        sensor_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Wake top 50% by energy
        num_to_wake = len(sensor_nodes) // 2
        wake_set = set(idx for idx, _ in sensor_nodes[:num_to_wake])
        
        for i in range(1, self.num_nodes):
            wake_list.append(1 if i in wake_set else 0)
        
        return wake_list
    
    def _policy_dqn(self, state: Dict) -> List[int]:
        """DQN policy: learned wake/sleep decisions."""
        state_vec = self.state_to_vector(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random action
            wake_probs = np.random.random(self.num_nodes - 1)
        else:
            # Use policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
                wake_probs = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # Convert probabilities to binary decisions (threshold at 0.5)
        wake_list = [1]  # Sink always on
        for prob in wake_probs:
            wake_list.append(1 if prob > 0.5 else 0)
        
        # Ensure at least some nodes are awake
        if sum(wake_list[1:]) < 3:
            # Wake 3 random nodes if too few
            indices = random.sample(range(1, self.num_nodes), 3)
            for i in indices:
                wake_list[i] = 1
        
        return wake_list
    
    def update(self, state: Dict, action: List[int], reward: float, next_state: Dict, done: bool):
        """Update DQN with experience."""
        if not self.dqn_enabled:
            return
        
        state_vec = self.state_to_vector(state)
        next_state_vec = self.state_to_vector(next_state)
        action_vec = np.array(action[1:], dtype=np.float32)  # Exclude sink
        
        # Store in replay buffer
        self.memory.push(state_vec, action_vec, reward, next_state_vec, done)
        
        # Train if enough samples
        if len(self.memory) >= BATCH_SIZE:
            self._train_step()
        
        # Update target network periodically
        if self.step % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def _train_step(self):
        """Perform one training step on a batch."""
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states)
            target_q = rewards.unsqueeze(1) + GAMMA_DISCOUNT * next_q * (1 - dones.unsqueeze(1))
        
        # MSE loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def step_episode(self, state: Dict) -> Tuple[List[int], float]:
        """Process one step: calculate reward, select action, update DQN."""
        self.step += 1
        
        # Calculate reward from previous action
        reward, packets, awake, dead = self.calculate_reward(state)
        self.total_reward += reward
        
        # Store metrics
        nodes = state.get("nodes", [])
        avg_energy = np.mean([n.get("energy", 0) for n in nodes[1:]])
        self.rewards_history.append(reward)
        self.packets_history.append(packets)
        self.energy_history.append(avg_energy)
        self.awake_history.append(awake)
        
        # Update DQN if we have previous state
        if self.prev_state is not None and self.dqn_enabled:
            self.update(self.prev_state, self.prev_action, reward, state, False)
        
        # Select next action
        action = self.select_action(state)
        
        # Store for next update
        self.prev_state = state
        self.prev_action = action
        
        return action, reward
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            "policy": self.policy,
            "step": self.step,
            "total_reward": self.total_reward,
            "avg_reward": np.mean(self.rewards_history[-100:]) if self.rewards_history else 0,
            "avg_packets": np.mean(self.packets_history[-100:]) if self.packets_history else 0,
            "avg_energy": np.mean(self.energy_history[-10:]) if self.energy_history else 0,
            "avg_awake": np.mean(self.awake_history[-10:]) if self.awake_history else 0,
            "epsilon": self.epsilon if self.dqn_enabled else 0,
        }


class WSNController:
    """Main controller that interfaces with ns-3 simulation."""
    
    def __init__(self, policy: str = "dqn", num_nodes: int = 25):
        self.agent = WSNAgent(num_nodes=num_nodes, policy=policy)
        
        # Setup UDP sockets
        self.rx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx_socket.bind((NS3_ADDR, NS3_RECV_PORT))
        self.rx_socket.settimeout(2.0)
        
        self.tx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        print(f"[Controller] Listening on {NS3_ADDR}:{NS3_RECV_PORT}")
        print(f"[Controller] Will send to {NS3_ADDR}:{NS3_SEND_PORT}")
    
    def send_action(self, wake_list: List[int]):
        """Send wake/sleep action to ns-3."""
        action_msg = json.dumps({"wake_list": wake_list})
        self.tx_socket.sendto(action_msg.encode(), (NS3_ADDR, NS3_SEND_PORT))
    
    def run(self, duration: int = 60):
        """Main control loop."""
        print(f"\n{'='*60}")
        print(f"  WSN-DQN Controller - Policy: {self.agent.policy.upper()}")
        print(f"  Running for {duration} seconds")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                try:
                    # Receive state from ns-3
                    data, addr = self.rx_socket.recvfrom(8192)
                    state = json.loads(data.decode())
                    
                    epoch = state.get("epoch", 0)
                    
                    # Agent decides action
                    action, reward = self.agent.step_episode(state)
                    
                    # Send action back to ns-3
                    self.send_action(action)
                    
                    # Print progress
                    stats = self.agent.get_stats()
                    awake_count = sum(action[1:])
                    print(f"[{epoch:5.1f}s] Reward: {reward:+6.1f} | "
                          f"Awake: {awake_count:2d}/24 | "
                          f"Energy: {stats['avg_energy']:5.1f}% | "
                          f"ε: {stats['epsilon']:.3f}")
                    
                except socket.timeout:
                    continue
                except json.JSONDecodeError as e:
                    print(f"[ERROR] JSON decode error: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\n[Controller] Interrupted by user")
        
        # Print final stats
        self._print_summary()
    
    def _print_summary(self):
        """Print final summary statistics."""
        stats = self.agent.get_stats()
        
        print(f"\n{'='*60}")
        print(f"  SIMULATION SUMMARY - {stats['policy'].upper()}")
        print(f"{'='*60}")
        print(f"  Total Steps:    {stats['step']}")
        print(f"  Total Reward:   {stats['total_reward']:.1f}")
        print(f"  Avg Reward:     {stats['avg_reward']:.2f}")
        print(f"  Avg Packets:    {stats['avg_packets']:.1f} per epoch")
        print(f"  Avg Energy:     {stats['avg_energy']:.1f}%")
        print(f"  Avg Awake:      {stats['avg_awake']:.1f} nodes")
        print(f"{'='*60}\n")
        
        return stats


def compare_policies(duration: int = 35):
    """Run comparison of different policies."""
    policies = ["always_on", "round_robin", "greedy_energy", "random"]
    if TORCH_AVAILABLE:
        policies.append("dqn")
    
    results = {}
    
    for policy in policies:
        print(f"\n{'#'*60}")
        print(f"  Testing Policy: {policy.upper()}")
        print(f"{'#'*60}")
        
        controller = WSNController(policy=policy)
        stats = controller.run(duration=duration)
        results[policy] = stats
        
        # Small delay between runs
        time.sleep(2)
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"{'POLICY COMPARISON':^70}")
    print(f"{'='*70}")
    print(f"{'Policy':<15} {'Total Reward':>12} {'Avg Packets':>12} {'Avg Energy':>12} {'Avg Awake':>10}")
    print(f"{'-'*70}")
    
    for policy, stats in results.items():
        print(f"{policy:<15} {stats['total_reward']:>12.1f} "
              f"{stats['avg_packets']:>12.1f} {stats['avg_energy']:>12.1f}% "
              f"{stats['avg_awake']:>10.1f}")
    
    print(f"{'='*70}\n")


def main():
    import sys
    
    print("\n" + "="*60)
    print("  WSN-DQN Agent")
    print("  Controls node wake/sleep scheduling to optimize")
    print("  throughput vs energy consumption")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python3 wsn_dqn_agent.py <policy> [duration]")
        print("\nPolicies:")
        print("  always_on     - All nodes always awake (baseline)")
        print("  round_robin   - Rotate which 50% are awake")
        print("  greedy_energy - Wake nodes with most energy")
        print("  random        - Randomly wake 50% each epoch")
        print("  dqn           - Deep Q-Network learned policy")
        print("  compare       - Compare all policies")
        print("\nExample:")
        print("  python3 wsn_dqn_agent.py dqn 60")
        print("  python3 wsn_dqn_agent.py compare")
        return
    
    policy = sys.argv[1].lower()
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 35
    
    if policy == "compare":
        compare_policies(duration)
    else:
        controller = WSNController(policy=policy)
        controller.run(duration=duration)


if __name__ == "__main__":
    main()
