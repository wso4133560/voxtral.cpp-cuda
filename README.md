# voxtral.cpp-cuda

> 基于 `ggml` 的 `Voxtral Realtime 4B` C++ 实现，支持本地离线转录、系统音频实时字幕、CUDA 加速，以及结合 Ollama 的实时中译工作流。

[![GitHub stars](https://img.shields.io/github/stars/wso4133560/voxtral.cpp-cuda?style=flat-square)](https://github.com/wso4133560/voxtral.cpp-cuda/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/wso4133560/voxtral.cpp-cuda?style=flat-square)](https://github.com/wso4133560/voxtral.cpp-cuda/network/members)
[![GitHub issues](https://img.shields.io/github/issues/wso4133560/voxtral.cpp-cuda?style=flat-square)](https://github.com/wso4133560/voxtral.cpp-cuda/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/wso4133560/voxtral.cpp-cuda?style=flat-square)](https://github.com/wso4133560/voxtral.cpp-cuda/commits)

如果你希望这个项目获得更大的曝光度，最直接的方式是 `Star`、`Fork`、转发项目链接，或者把它分享给正在做本地语音识别、实时字幕、CUDA 推理、Ollama 工作流的开发者。

这个仓库重点提供：
- 本地离线语音转文本能力
- 基于 PulseAudio 的系统音频实时转录
- CUDA 加速推理
- 结合 Ollama 的实时中译能力
- GGUF 模型量化与推理工作流

如果你正在寻找可本地部署的 C++ 实时语音识别方案，或者希望把会议、直播、视频、系统音频实时转成字幕，这个项目就是为这些场景准备的。

## 性能速览

以下数据来自仓库内的本地测试记录，测试时间为 **2026-03-16**，硬件为 **RTX 3080**，模型为 **Voxtral Realtime 4B Q4_0**：

| 指标 | 结果 |
|------|------|
| 优化后吞吐量 | **170.7 tok/s** |
| 干净环境平均性能 | **168.84 tok/s** |
| 相比基线提升 | **+30.5%** |
| KV cache 显存占用 | **-50%** |
| WER | **0.0000** |
| 实时性 | **RTF 0.086** |

相关文档：
- [OPTIMIZATION_REPORT.md](/home/tanglin/workspace2/voxtral.cpp-cuda/OPTIMIZATION_REPORT.md)
- [FINAL_PERFORMANCE_REPORT.md](/home/tanglin/workspace2/voxtral.cpp-cuda/FINAL_PERFORMANCE_REPORT.md)
- [REALTIME_TRANSCRIPTION.md](/home/tanglin/workspace2/voxtral.cpp-cuda/REALTIME_TRANSCRIPTION.md)

## 项目亮点

- **纯本地部署**：适合对隐私、延迟和可控性有要求的场景
- **C++ + ggml**：部署简单，便于嵌入现有本地推理栈
- **支持 CUDA**：在 NVIDIA GPU 上获得更快推理速度
- **实时转录**：可直接捕获系统音频并在终端输出实时文本
- **实时翻译**：可将多种语言实时转录结果继续交给本地 Ollama 模型翻译为中文
- **支持量化**：便于在不同硬件资源下平衡速度与精度

## 为什么值得关注

- **不是演示级原型**：仓库里包含 CUDA 优化、实时转录、实时翻译和量化工具链，不只是单次离线推理
- **适合二次开发**：如果你想把语音识别接入本地助手、字幕系统、会议记录或内容处理流水线，这个仓库是一个直接可改的 C++ 基础设施
- **有明确性能结果**：首页即可看到关键性能指标，降低使用者的评估成本
- **传播门槛低**：项目定位清晰，目标用户明确，适合在 GitHub、技术社区、中文开发者社群中快速传播

## 一句话介绍

`voxtral.cpp-cuda` 是一个面向本地部署的 C++ 实时语音识别项目，支持 CUDA 加速、系统音频实时转录，以及结合 Ollama 的实时中文翻译。

## 适用场景

- 会议记录与字幕生成
- 直播、视频、课程音频监听与转写
- 本地 AI 语音助手前端输入
- 多种语言内容的实时中文字幕辅助
- 需要离线语音识别能力的 Linux / C++ 项目集成

## 快速开始

### 1. 下载模型

从 Hugging Face 下载已经转换好的 GGUF 模型：

```bash
# 默认下载 Q4_0 量化版本
./tools/download_model.sh Q4_0
```

### 2. 编译项目

使用 CMake 编译：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

如果你希望启用 CUDA，可在配置时加上：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j
```

### 3. 准备音频

模型输入要求为 **16-bit PCM WAV**，采样率 **16kHz 单声道**。可以使用 `ffmpeg` 转换：

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

### 4. 运行推理

```bash
./build/voxtral \
  --model models/voxtral/Q4_0.gguf \
  --audio path/to/input.wav \
  --threads 8
```

---

## 进阶用法

### 优化版实时转录

如果编译时检测到 PulseAudio，CMake 会额外生成优化版实时转录客户端：

```bash
./build/voxtral-realtime-opt \
  --model models/voxtral/Q4_0.gguf \
  --cuda \
  --threads 8 \
  --interval 1500 \
  --show-stats
```

开始前可以先查看可用的 PulseAudio 采集源：

```bash
./build/voxtral-realtime-opt --list-sources
```

### 实时中文翻译

如果编译时检测到 PulseAudio，CMake 也会生成一个实时翻译封装程序。它会把 `voxtral-realtime-opt` 输出的多种语言实时转录结果继续发送给本地 Ollama 翻译模型，实现实时中译。

先准备 Ollama 模型：

```bash
ollama pull translategemma:4b-it-q4_K_M
```

`voxtral-realtime-translate` 默认使用的翻译模型为：
- `translategemma:4b-it-q4_K_M`

运行实时中文翻译客户端：

```bash
./build/voxtral-realtime-translate \
  --model models/voxtral/Q4_0.gguf \
  --cuda \
  --show-original
```

如果需要，也可以手动指定 Ollama 地址或模型名称：

```bash
./build/voxtral-realtime-translate \
  --model models/voxtral/Q4_0.gguf \
  --cuda \
  --ollama-host 127.0.0.1:11434 \
  --ollama-model translategemma:4b-it-q4_K_M
```

### 手动量化

你也可以使用原生量化工具对现有 GGUF 模型继续量化：

```bash
./build/voxtral-quantize \
  models/voxtral/voxtral.gguf \
  models/voxtral/voxtral-q6_k.gguf \
  Q6_K \
  8
```

## 测试

测试会基于 `samples/*.wav` 运行。

### 数值一致性检查

用于验证与参考实现的数值一致性：

```bash
python3 tests/test_voxtral_reference.py
```

### 自定义容差

你可以通过环境变量覆盖比较容差：
- `VOXTRAL_TEST_ATOL`（默认：`1e-2`）
- `VOXTRAL_TEST_RTOL`（默认：`1e-2`）
- `VOXTRAL_TEST_THREADS`

## 欢迎支持

如果这个项目对你有帮助，欢迎：
- Star 本仓库
- Fork 并提交改进
- 提交 Issue / PR 一起完善功能
- 分享给对本地语音识别、实时字幕、CUDA 推理、Ollama 翻译感兴趣的朋友

这些支持都能帮助项目获得更大的曝光度，也能让更多需要本地实时语音能力的开发者找到它。
