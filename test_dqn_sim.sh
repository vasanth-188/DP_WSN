#!/bin/bash
# Test script for DQN WSN simulation
# Run ns-3 simulation and Python agent together

echo "=== WSN DQN Simulation Test ==="
echo ""

# Start ns-3 simulation in background
echo "[1] Starting ns-3 simulation..."
cd /Users/sham/Desktop/ns-3.46.1
./ns3 run scratch/wsn_dqn 2>&1 &
NS3_PID=$!
sleep 2

# Start Python DQN agent
echo "[2] Starting DQN agent..."
python3 wsn_dqn_agent.py --policy round_robin --episodes 1 &
PYTHON_PID=$!

# Wait for both to complete
wait $NS3_PID
wait $PYTHON_PID

echo ""
echo "=== Test Complete ==="
echo "Animation saved to: wsn_dqn_anim.xml"
