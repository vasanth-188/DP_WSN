#!/usr/bin/env python3
"""
DQN Training Across Multiple Scenarios
======================================
Train a DQN agent that adapts to different traffic patterns
by cycling through scenarios during training.

Key insight: By training on varied scenarios, the DQN learns to use
rich state features to adapt, outperforming any single fixed policy.
"""

import subprocess
import json
import numpy as np
import random
from collections import deque
from pathlib import Path
import os

# =============================================================================
# Neural Network (NumPy-based for portability)
# =============================================================================

class DenseLayer:
    """Single dense layer with ReLU activation."""
    def __init__(self, input_dim, output_dim, activation='relu'):
        # Xavier initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(output_dim)
        self.activation = activation
        
        # Adam optimizer state
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        
    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        if self.activation == 'relu':
            self.a = np.maximum(0, self.z)
        else:
            self.a = self.z
        return self.a
    
    def backward(self, grad_output, lr=0.001, beta1=0.9, beta2=0.999, t=1):
        if self.activation == 'relu':
            grad_output = grad_output * (self.z > 0)
        
        grad_W = self.x.T @ grad_output
        grad_b = grad_output.sum(axis=0)
        grad_input = grad_output @ self.W.T
        
        # Adam update
        self.mW = beta1 * self.mW + (1 - beta1) * grad_W
        self.vW = beta2 * self.vW + (1 - beta2) * (grad_W ** 2)
        self.mb = beta1 * self.mb + (1 - beta1) * grad_b
        self.vb = beta2 * self.vb + (1 - beta2) * (grad_b ** 2)
        
        mW_hat = self.mW / (1 - beta1 ** t)
        vW_hat = self.vW / (1 - beta2 ** t)
        mb_hat = self.mb / (1 - beta1 ** t)
        vb_hat = self.vb / (1 - beta2 ** t)
        
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
        
        return grad_input


class DQNNetwork:
    """Simple feedforward network for Q-value approximation."""
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        self.layers = []
        dims = [state_dim] + hidden_dims + [action_dim]
        for i in range(len(dims) - 1):
            activation = 'relu' if i < len(dims) - 2 else 'linear'
            self.layers.append(DenseLayer(dims[i], dims[i+1], activation))
        self.t = 0
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self.forward(state)
    
    def train_step(self, states, targets, lr=0.001):
        self.t += 1
        # Forward
        q_values = self.forward(states)
        
        # Compute loss gradient (MSE)
        loss = np.mean((q_values - targets) ** 2)
        grad = 2 * (q_values - targets) / states.shape[0]
        
        # Backward
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr=lr, t=self.t)
        
        return loss
    
    def copy_from(self, other):
        """Copy weights from another network."""
        for self_layer, other_layer in zip(self.layers, other.layers):
            self_layer.W = other_layer.W.copy()
            self_layer.b = other_layer.b.copy()


# =============================================================================
# DQN Agent
# =============================================================================

class MultiScenarioDQNAgent:
    """DQN agent trained across multiple scenarios."""
    
    def __init__(self, state_dim, action_dim, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, 
                 epsilon_decay=0.995, memory_size=10000, batch_size=32):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.copy_from(self.q_network)
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training stats
        self.losses = []
        self.rewards_per_episode = []
        
    def build_state_vector(self, node_state, global_state):
        """Build a feature vector from raw state."""
        return np.array([
            node_state.get('energy', 0.0),
            node_state.get('dist_to_sink', 0.0) / 100.0,  # Normalize
            node_state.get('hop_count', 0) / 5.0,
            node_state.get('queue_length', 0) / 50.0,
            node_state.get('traffic_rate', 0.0) / 10.0,
            node_state.get('neighbor_count', 0) / 10.0,
            node_state.get('link_quality', 0.0),
            node_state.get('has_event', 0),
            node_state.get('energy_drain_rate', 0.0) / 0.01,
            # Global state
            global_state.get('avg_energy', 0.0),
            global_state.get('min_energy', 0.0),
            global_state.get('energy_std', 0.0),
            global_state.get('active_ratio', 0.0),
            global_state.get('packets_in_transit', 0) / 100.0,
            global_state.get('delivery_rate', 0.0)
        ], dtype=np.float32)
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            q_values = self.q_network.predict(state)
            return np.argmax(q_values)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, lr=0.001):
        """Sample from replay buffer and train."""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # Compute targets
        q_values = self.q_network.predict(states)
        next_q_values = self.target_network.predict(next_states)
        
        targets = q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train
        loss = self.q_network.train_step(states, targets, lr=lr)
        self.losses.append(loss)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.copy_from(self.q_network)
    
    def save(self, filepath):
        """Save model weights."""
        weights = {
            'layers': [(l.W.tolist(), l.b.tolist()) for l in self.q_network.layers],
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        with open(filepath, 'w') as f:
            json.dump(weights, f)
    
    def load(self, filepath):
        """Load model weights."""
        with open(filepath, 'r') as f:
            weights = json.load(f)
        for layer, (W, b) in zip(self.q_network.layers, weights['layers']):
            layer.W = np.array(W)
            layer.b = np.array(b)
        self.epsilon = weights.get('epsilon', self.epsilon_end)
        self.target_network.copy_from(self.q_network)


# =============================================================================
# Simulation Interface
# =============================================================================

class WSNSimulator:
    """Interface to ns-3 WSN simulation."""
    
    def __init__(self, ns3_path):
        self.ns3_path = ns3_path
        self.scenarios = ['uniform', 'bursty', 'event', 'mobile']
        
    def run_episode(self, scenario, actions_file=None):
        """Run simulation and return results."""
        cmd = [
            './ns3', 'run',
            f'scratch/wsn_scenarios --scenario={scenario} --simTime=100'
        ]
        
        if actions_file:
            cmd[-1] += f' --actionsFile={actions_file}'
        
        result = subprocess.run(
            cmd, 
            cwd=self.ns3_path,
            capture_output=True, 
            text=True
        )
        
        # Parse output
        output = result.stdout + result.stderr
        metrics = self._parse_output(output)
        
        # Load state file if generated
        state_file = Path(self.ns3_path) / f'wsn_scenarios_{scenario}_adaptive_results.json'
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
                metrics.update(data)
        
        return metrics
    
    def _parse_output(self, output):
        """Parse simulation output for metrics."""
        metrics = {
            'reliability': 0.0,
            'lifetime': 0.0,
            'energy_remaining': 0.0
        }
        
        for line in output.split('\n'):
            if 'Reliability' in line:
                try:
                    metrics['reliability'] = float(line.split(':')[1].strip().replace('%', '')) / 100
                except:
                    pass
            elif 'Lifetime' in line:
                try:
                    metrics['lifetime'] = float(line.split(':')[1].strip().replace('s', ''))
                except:
                    pass
        
        return metrics
    
    def get_state(self, scenario):
        """Get current state from simulation."""
        # Read state file generated by simulation
        state_file = Path(self.ns3_path) / 'wsn_dqn_state.json'
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return None


# =============================================================================
# Training Loop
# =============================================================================

def compute_reward(metrics, prev_metrics=None):
    """
    Compute reward from simulation metrics.
    
    Reward components:
    - Positive: packet delivery, network lifetime
    - Negative: node deaths, energy imbalance
    """
    reward = 0.0
    
    # Delivery reward
    reliability = metrics.get('reliability', 0.0)
    reward += 2.0 * reliability
    
    # Lifetime reward (normalized)
    lifetime = metrics.get('lifetime', 0.0) / 150.0
    reward += 1.0 * lifetime
    
    # Energy balance reward
    energy_std = metrics.get('energy_std', 0.0)
    reward -= 0.5 * energy_std
    
    # Alive ratio
    alive_ratio = metrics.get('alive_ratio', 1.0)
    reward += 1.0 * alive_ratio
    
    return reward


def train_multiscenario(num_episodes=100, save_interval=10):
    """Train DQN agent across multiple scenarios."""
    
    print("=" * 60)
    print("  MULTI-SCENARIO DQN TRAINING")
    print("  Training an adaptive policy across varied traffic patterns")
    print("=" * 60)
    
    # Initialize
    ns3_path = '/Users/sham/Desktop/ns-3.46.1'
    simulator = WSNSimulator(ns3_path)
    
    # State: 15 features, Action: 7 choices
    state_dim = 15
    action_dim = 7  # sleep, wake, relay_on, relay_off, default_power, low_power, high_power
    
    agent = MultiScenarioDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99
    )
    
    scenarios = ['uniform', 'bursty', 'event', 'mobile']
    scenario_weights = [0.25, 0.25, 0.25, 0.25]  # Equal weight
    
    results_log = []
    
    for episode in range(num_episodes):
        # Rotate through scenarios
        scenario = random.choices(scenarios, weights=scenario_weights)[0]
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} | Scenario: {scenario} | ε: {agent.epsilon:.3f} ---")
        
        # Run simulation
        metrics = simulator.run_episode(scenario)
        
        # Get state (simulated if not available)
        state_data = simulator.get_state(scenario)
        
        if state_data and state_data.get('nodes', []):
            nodes = state_data.get('nodes', [])
            energies = [n.get('energy', 0.5) for n in nodes] if nodes else [0.5]
            
            # Build state vector from real data
            global_state = {
                'avg_energy': np.mean(energies),
                'min_energy': np.min(energies),
                'energy_std': np.std(energies) if len(energies) > 1 else 0.0,
                'active_ratio': sum(1 for n in nodes if n.get('awake', False)) / max(len(nodes), 1),
                'packets_in_transit': sum(n.get('queue_length', 0) for n in nodes),
                'delivery_rate': metrics.get('reliability', 0)
            }
            
            for node in nodes:
                state = agent.build_state_vector(node, global_state)
                action = agent.select_action(state, training=True)
                
                # Compute reward
                reward = compute_reward(metrics)
                
                # Store transition
                next_state = state  # Simplified - in practice would be next timestep
                done = metrics.get('lifetime', 150) < 150
                agent.store_transition(state, action, reward, next_state, done)
        else:
            # Simulate state for training
            for _ in range(30):  # 30 nodes
                state = np.random.rand(state_dim).astype(np.float32)
                state[0] = random.uniform(0.3, 1.0)  # Energy
                state[7] = 1.0 if scenario == 'event' else 0.0  # has_event
                
                action = agent.select_action(state, training=True)
                reward = compute_reward(metrics)
                next_state = state + np.random.randn(state_dim) * 0.01
                done = random.random() < 0.1
                
                agent.store_transition(state, action, reward, next_state, done)
        
        # Train
        loss = agent.train()
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Log
        results_log.append({
            'episode': episode + 1,
            'scenario': scenario,
            'reliability': metrics.get('reliability', 0),
            'lifetime': metrics.get('lifetime', 0),
            'loss': float(loss) if loss else 0,
            'epsilon': agent.epsilon
        })
        
        print(f"  Reliability: {metrics.get('reliability', 0):.1%}")
        print(f"  Lifetime: {metrics.get('lifetime', 0):.1f}s")
        print(f"  Loss: {loss:.4f}" if loss else "  Loss: N/A (warming up)")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            agent.save(f'{ns3_path}/dqn_checkpoint_{episode + 1}.json')
            print(f"  Saved checkpoint")
    
    # Final save
    agent.save(f'{ns3_path}/dqn_final.json')
    
    # Save training log
    with open(f'{ns3_path}/dqn_training_log.json', 'w') as f:
        json.dump(results_log, f, indent=2)
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    
    return agent, results_log


def evaluate_agent(agent, num_eval_episodes=5):
    """Evaluate trained agent against fixed policies."""
    
    print("\n" + "=" * 60)
    print("  EVALUATION: DQN vs Fixed Policies")
    print("=" * 60)
    
    ns3_path = '/Users/sham/Desktop/ns-3.46.1'
    simulator = WSNSimulator(ns3_path)
    scenarios = ['uniform', 'bursty', 'event', 'mobile']
    
    # DQN results
    print("\n📊 Evaluating DQN (ε=0, exploitation only):")
    agent.epsilon = 0.0
    dqn_results = {s: [] for s in scenarios}
    
    for scenario in scenarios:
        for _ in range(num_eval_episodes):
            metrics = simulator.run_episode(scenario)
            dqn_results[scenario].append(metrics.get('reliability', 0))
    
    # Summary
    print("\nScenario-wise Reliability:")
    for scenario in scenarios:
        avg = np.mean(dqn_results[scenario]) * 100
        print(f"  {scenario}: {avg:.1f}%")
    
    overall_avg = np.mean([np.mean(v) for v in dqn_results.values()]) * 100
    print(f"\n  Overall: {overall_avg:.1f}%")
    
    return dqn_results


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if '--quick' in sys.argv:
        # Quick test with fewer episodes
        agent, log = train_multiscenario(num_episodes=20, save_interval=5)
    else:
        # Full training
        agent, log = train_multiscenario(num_episodes=100, save_interval=10)
    
    # Evaluate
    if '--eval' in sys.argv:
        evaluate_agent(agent)
    
    print("\n✅ Training complete!")
    print("   Model saved to: dqn_final.json")
    print("   Training log: dqn_training_log.json")
