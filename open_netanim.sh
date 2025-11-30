#!/bin/bash
# open_netanim.sh - Launch NetAnim with wsn_dqn_anim.xml

NETANIM_BIN="/Users/sham/Desktop/ns-3.46.1/ns-allinone-3.42/netanim-3.109/NetAnim"
ANIMATION_FILE="/Users/sham/Desktop/ns-3.46.1/wsn_dqn_anim.xml"

if [ ! -f "$NETANIM_BIN" ]; then
    echo "Error: NetAnim binary not found at $NETANIM_BIN"
    exit 1
fi

if [ ! -f "$ANIMATION_FILE" ]; then
    echo "Error: Animation file not found at $ANIMATION_FILE"
    echo "Did you run: ./ns3 run wsn_dqn?"
    exit 1
fi

echo "Launching NetAnim with $ANIMATION_FILE..."
"$NETANIM_BIN" "$ANIMATION_FILE" &
NETANIM_PID=$!
echo "NetAnim started (PID: $NETANIM_PID)"
echo "Tip: File > Open to load different animation files"
