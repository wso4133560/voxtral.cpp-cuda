#!/bin/bash

# Real-time Audio Transcription Demo
# This script demonstrates the voxtral-realtime client

MODEL="models/voxtral/Q4_0.gguf"
VOXTRAL_RT="build/voxtral-realtime"

echo "=========================================="
echo "Real-time Audio Transcription Demo"
echo "=========================================="
echo ""

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    echo "Please ensure the model file exists."
    exit 1
fi

# Check if voxtral-realtime exists
if [ ! -f "$VOXTRAL_RT" ]; then
    echo "Error: voxtral-realtime not found at $VOXTRAL_RT"
    echo "Please build the project first."
    exit 1
fi

echo "Available audio sources:"
echo "------------------------"
pactl list sources short
echo ""

echo "Starting real-time transcription..."
echo "The client will capture system audio and display transcriptions."
echo ""
echo "Tips:"
echo "  - Play any audio/video on your system"
echo "  - The transcription will appear in real-time"
echo "  - Press Ctrl+C to stop"
echo ""
echo "=========================================="
echo ""

# Run with CUDA backend
$VOXTRAL_RT --model "$MODEL" --cuda --interval 2000

echo ""
echo "=========================================="
echo "Transcription stopped."
echo "=========================================="
