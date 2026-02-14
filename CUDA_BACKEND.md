# CUDA Backend Support for Voxtral.cpp

## Overview

This project now supports NVIDIA CUDA backend for GPU acceleration, in addition to the existing CPU and Metal backends.

## Features

- **GPU Acceleration**: Leverage NVIDIA GPUs for faster inference
- **Automatic Fallback**: CPU backend is used as fallback for unsupported operations
- **Backend-Agnostic KV Cache**: Properly handles KV cache operations across different backends
- **Easy to Use**: Simple command-line flag to enable CUDA

## Requirements

- NVIDIA GPU with CUDA support (Compute Capability 5.0+)
- CUDA Toolkit 11.0 or later
- CMake 3.16+
- GCC/Clang compiler

## Building with CUDA Support

### 1. Clean build directory (if exists)
```bash
rm -rf build
mkdir build
cd build
```

### 2. Configure with CUDA enabled
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
```

### 3. Build the project
```bash
cmake --build . -j$(nproc)
```

## Usage

### Basic Usage with CUDA

```bash
./voxtral --model path/to/model.gguf --audio path/to/audio.wav --cuda
```

### Command-Line Options

- `--cuda`: Enable CUDA backend for GPU acceleration
- `--metal`: Enable Metal backend (macOS only)
- `--threads N`: Number of CPU threads for fallback operations

### Examples

#### Transcribe with CUDA
```bash
./build/voxtral --model models/voxtral/Q4_0.gguf --audio samples/8297-275156-0000.wav --cuda
```

#### Transcribe with CPU (default)
```bash
./build/voxtral --model models/voxtral/Q4_0.gguf --audio samples/8297-275156-0000.wav
```

## Testing

A test script is provided to verify CUDA backend functionality:

```bash
./test_cuda.sh
```

This script tests all sample audio files and verifies the transcription output.

## Performance

CUDA backend provides significant speedup for:
- Encoder operations (mel spectrogram processing)
- Attention mechanisms
- Matrix multiplications
- Decoder inference

Expected speedup: 3-10x compared to CPU, depending on GPU model and batch size.

## Supported GPUs

The build automatically detects and compiles for multiple GPU architectures:
- Maxwell (Compute Capability 5.0)
- Pascal (Compute Capability 6.1)
- Volta (Compute Capability 7.0)
- Turing (Compute Capability 7.5)
- Ampere (Compute Capability 8.0, 8.6)
- Ada Lovelace (Compute Capability 8.9)

## Implementation Details

### Backend Selection Priority

1. **CUDA** (if `--cuda` flag is provided and CUDA is available)
2. **Metal** (if `--metal` flag is provided and Metal is available, macOS only)
3. **CPU** (default fallback)

### Key Changes

1. **CMakeLists.txt**: Added `GGML_CUDA` option
2. **voxtral.cpp**:
   - Added CUDA backend initialization
   - Implemented backend-agnostic KV cache operations
   - Added `voxtral_model_load_from_file_ex()` function with CUDA support
3. **voxtral.h**: Added `use_cuda` parameter to context params
4. **main.cpp**: Added `--cuda` command-line option

### KV Cache Handling

The KV cache operations have been updated to work with GPU backends:
- `clear_kv_cache()`: Uses `ggml_backend_tensor_set()` instead of direct `memset()`
- `kv_cache_shift_left()`: Transfers data to CPU, performs shift, then transfers back

This ensures compatibility with both CPU and GPU memory spaces.

## Troubleshooting

### CUDA not detected
```
voxtral: CUDA backend not available in this build, using CPU
```
**Solution**: Rebuild with `-DGGML_CUDA=ON` flag

### Out of memory
```
ggml_cuda_init: failed to allocate memory
```
**Solution**:
- Use a smaller model (e.g., Q4_0 instead of F16)
- Close other GPU-intensive applications
- Check available GPU memory with `nvidia-smi`

### Segmentation fault
If you encounter crashes, ensure:
- CUDA drivers are up to date
- GPU has sufficient memory
- Model file is not corrupted

## Benchmarking

To compare CPU vs CUDA performance:

```bash
# CPU benchmark
time ./build/voxtral --model models/voxtral/Q4_0.gguf --audio samples/8297-275156-0002.wav

# CUDA benchmark
time ./build/voxtral --model models/voxtral/Q4_0.gguf --audio samples/8297-275156-0002.wav --cuda
```

## Contributing

When contributing CUDA-related changes:
1. Ensure backward compatibility with CPU backend
2. Test on multiple GPU architectures if possible
3. Update this documentation
4. Run the test suite: `./test_cuda.sh`

## License

Same as the main project license.
