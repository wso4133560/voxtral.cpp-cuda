#!/bin/bash

# Test real-time transcription with sample audio
# This simulates real-time playback by streaming a WAV file to PulseAudio

MODEL="models/voxtral/Q4_0.gguf"
SAMPLE="samples/8297-275156-0002.wav"
VOXTRAL_RT="build/voxtral-realtime"

echo "=========================================="
echo "Real-time Transcription Test"
echo "=========================================="
echo ""

# Check if files exist
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    exit 1
fi

if [ ! -f "$SAMPLE" ]; then
    echo "Error: Sample audio not found at $SAMPLE"
    exit 1
fi

if [ ! -f "$VOXTRAL_RT" ]; then
    echo "Error: voxtral-realtime not found"
    exit 1
fi

echo "This test will:"
echo "1. Start the real-time transcription client"
echo "2. Play a sample audio file through your system"
echo "3. Show live transcription in the terminal"
echo ""
echo "Expected output:"
echo "  'We have both seen the same newspaper...'"
echo ""
echo "Press Enter to start..."
read

# Start transcription client in background
echo "Starting transcription client..."
$VOXTRAL_RT --model "$MODEL" --cuda --interval 1500 &
RT_PID=$!

# Give it time to initialize
sleep 3

echo ""
echo "Playing sample audio..."
echo "=========================================="

# Play the audio file
if command -v paplay &> /dev/null; then
    paplay "$SAMPLE"
elif command -v aplay &> /dev/null; then
    aplay "$SAMPLE"
else
    echo "Error: No audio player found (paplay or aplay)"
    kill $RT_PID
    exit 1
fi

# Wait a bit for transcription to complete
sleep 5

# Stop the transcription client
echo ""
echo "=========================================="
echo "Stopping transcription client..."
kill $RT_PID 2>/dev/null
wait $RT_PID 2>/dev/null

echo ""
echo "Test completed!"
echo ""
echo "If you saw the transcription above, the real-time client is working!"
echo "You can now use it with any system audio by running:"
echo "  $VOXTRAL_RT --model $MODEL --cuda"
