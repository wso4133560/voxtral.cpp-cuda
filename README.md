# voxtral.cpp

A ggml-based C++ implementation of Voxtral Realtime 4B.

## Quickstart

### 1. Download the model

Download the pre-converted GGUF model from Hugging Face:

```bash
# Default: Q4_0 quantization
./tools/download_model.sh Q4_0
```

### 2. Build

Build the project using CMake:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3. Audio Preparation

The model expects **16-bit PCM WAV** files at **16kHz (mono)**. You can use `ffmpeg` to convert your audio files:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

### 4. Run Inference

```bash
./build/voxtral \
  --model models/voxtral/Q4_0.gguf \
  --audio path/to/input.wav \
  --threads 8
```

---

## Advanced Usage

### Optimized Real-time Transcription

If PulseAudio is available during the build, CMake also generates an optimized real-time client:

```bash
./build/voxtral-realtime-opt \
  --model models/voxtral/Q4_0.gguf \
  --cuda \
  --threads 8 \
  --interval 1500 \
  --show-stats
```

To inspect available PulseAudio capture sources before starting:

```bash
./build/voxtral-realtime-opt --list-sources
```

### Real-time Chinese Translation

If PulseAudio is available during the build, CMake also generates a translation wrapper that pipes
the live English transcript from `voxtral-realtime-opt` through a local Ollama translation model.

Prepare the Ollama model first:

```bash
ollama pull translategemma:4b-it-q4_K_M
```

The default translation model used by `voxtral-realtime-translate` is:
- `translategemma:4b-it-q4_K_M`

Run the real-time Chinese translation client:

```bash
./build/voxtral-realtime-translate \
  --model models/voxtral/Q4_0.gguf \
  --cuda \
  --show-original
```

If needed, you can override the Ollama endpoint or model name:

```bash
./build/voxtral-realtime-translate \
  --model models/voxtral/Q4_0.gguf \
  --cuda \
  --ollama-host 127.0.0.1:11434 \
  --ollama-model translategemma:4b-it-q4_K_M
```

### Manual Quantization

You can quantize an existing GGUF file using the native quantizer:

```bash
./build/voxtral-quantize \
  models/voxtral/voxtral.gguf \
  models/voxtral/voxtral-q6_k.gguf \
  Q6_K \
  8
```

## Testing

The test suite runs over `samples/*.wav` files.

### Numeric Parity Check

To verify numeric parity against the reference implementation:

```bash
python3 tests/test_voxtral_reference.py
```

### Custom Tolerances

You can override comparison tolerances via environment variables:
- `VOXTRAL_TEST_ATOL` (default: 1e-2)
- `VOXTRAL_TEST_RTOL` (default: 1e-2)
- `VOXTRAL_TEST_THREADS`
