#!/bin/bash

# Test CUDA backend with all samples

MODEL="models/voxtral/Q4_0.gguf"
SAMPLES_DIR="samples"
VOXTRAL_BIN="build/voxtral"

echo "=========================================="
echo "Testing CUDA Backend"
echo "=========================================="
echo ""

for sample in 8297-275156-0000 8297-275156-0001 8297-275156-0002; do
    echo "Testing sample: $sample"
    echo "------------------------------------------"

    # Run with CUDA and extract only the transcription (first line of stdout)
    output=$($VOXTRAL_BIN --model "$MODEL" --audio "$SAMPLES_DIR/${sample}.wav" --cuda 2>/dev/null | head -1)
    expected=$(cat "$SAMPLES_DIR/${sample}.txt")

    # Normalize: lowercase, remove punctuation, trim whitespace
    output_norm=$(echo "$output" | tr '[:upper:]' '[:lower:]' | tr -d '[:punct:]' | xargs)
    expected_norm=$(echo "$expected" | tr '[:upper:]' '[:lower:]' | tr -d '[:punct:]' | xargs)

    echo "Expected: $expected"
    echo "Got:      $output"

    if [ "$output_norm" = "$expected_norm" ]; then
        echo "✓ PASSED"
    else
        echo "✗ FAILED"
        echo "  Normalized expected: $expected_norm"
        echo "  Normalized got:      $output_norm"
    fi
    echo ""
done

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
