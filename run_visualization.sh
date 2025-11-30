#!/bin/bash
# run_visualization.sh
# Complete setup script for WSN-DQN simulation with both visualizations

set -e

WORKSPACE="/Users/sham/Desktop/ns-3.46.1"
cd "$WORKSPACE"

echo "================================"
echo "WSN-DQN Visualization Setup"
echo "================================"
echo ""

# Check for matplotlib
echo "[1/4] Checking dependencies..."
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "    ⚠️  matplotlib not installed"
    echo "    Install with: pip3 install matplotlib"
    echo "    Continuing without live plotting..."
    ENABLE_PLOTTING=false
else
    echo "    ✓ matplotlib available"
    ENABLE_PLOTTING=true
fi
echo ""

# Check for ns-3
if [ ! -f "./ns3" ]; then
    echo "    ✗ ns3 executable not found at ./ns3"
    exit 1
fi
echo "    ✓ ns-3 build system ready"
echo ""

# Build the project
echo "[2/4] Building ns-3 project..."
./ns3 build > /dev/null 2>&1 || {
    echo "    ✗ Build failed"
    exit 1
}
echo "    ✓ Build successful"
echo ""

# Start ns-3 simulation
echo "[3/4] Starting ns-3 simulation (PID: $$)..."
echo "    Press Ctrl+C in Terminal 2 to stop"
./ns3 run wsn_dqn &
NS3_PID=$!
echo "    ✓ Simulation running (PID $NS3_PID)"
echo ""

# Start Python bridge
echo "[4/4] Starting Python bridge..."
if [ "$ENABLE_PLOTTING" = true ]; then
    echo "    Live plotting enabled"
    python3 wsn_bridge.py
else
    echo "    Live plotting disabled (use --no-plot flag)"
    python3 wsn_bridge.py --no-plot
fi

# Cleanup
wait $NS3_PID 2>/dev/null || true

echo ""
echo "================================"
echo "Simulation Complete!"
echo "================================"
echo ""
echo "Output files created:"
echo "  - wsn_dqn_anim.xml    : NetAnim animation file"
echo "  - wsn_monitor.png     : Final plot snapshot"
echo "  - wsn_history.json    : Raw simulation data"
echo ""
echo "Next steps:"
echo "  1. View offline animation:"
echo "     NetAnim wsn_dqn_anim.xml"
echo "  2. Analyze data:"
echo "     python3 -c \"import json; d=json.load(open('wsn_history.json')); print(d.keys())\""
echo ""
