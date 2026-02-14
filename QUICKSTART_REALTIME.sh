#!/bin/bash

cat << 'EOF'
========================================
Voxtral Real-time Transcription
Quick Start Guide
========================================

🎉 实时音频转录客户端已成功构建！

📋 功能特性
-----------
✓ 捕获系统音频（任何正在播放的音频/视频）
✓ 实时转录并显示在终端
✓ 支持CUDA GPU加速
✓ 低延迟（约1.5-2秒）
✓ 自动过滤静音

🚀 快速开始
-----------

1. 列出可用的音频源：
   ./build/voxtral-realtime --list-sources

2. 开始实时转录（使用默认音频监视器）：
   ./build/voxtral-realtime --model models/voxtral/Q4_0.gguf --cuda

3. 播放任何音频/视频，观察实时转录！

📖 使用示例
-----------

# 基本使用（CUDA加速）
./build/voxtral-realtime --model models/voxtral/Q4_0.gguf --cuda

# 更快的响应（1秒间隔）
./build/voxtral-realtime --model models/voxtral/Q4_0.gguf --cuda --interval 1000

# 指定特定音频源
./build/voxtral-realtime --model models/voxtral/Q4_0.gguf \
    --source alsa_output.pci-0000_00_1f.3.analog-stereo.monitor \
    --cuda

# 保存转录到文件
./build/voxtral-realtime --model models/voxtral/Q4_0.gguf --cuda 2>/dev/null > transcript.txt

🎬 运行演示
-----------
./demo_realtime.sh

🧪 运行测试
-----------
./test_realtime.sh

📚 详细文档
-----------
- REALTIME_TRANSCRIPTION.md - 完整使用指南
- REALTIME_IMPLEMENTATION_SUMMARY.md - 实现细节

⚙️ 命令行选项
-------------
--model PATH      模型文件路径（必需）
--source NAME     音频源名称（可选，默认使用系统监视器）
--interval MS     转录间隔（毫秒，默认2000）
--threads N       CPU线程数（默认4）
--cuda            使用CUDA加速
--metal           使用Metal加速（macOS）
--list-sources    列出可用音频源
-h, --help        显示帮助

💡 提示
-------
• 按 Ctrl+C 停止转录
• 使用 --cuda 获得最佳性能
• 调整 --interval 平衡响应速度和准确度
• 确保系统音量足够大以被检测到

🎯 使用场景
-----------
✓ YouTube视频实时字幕
✓ 在线会议转录
✓ 电影/电视剧字幕
✓ 播客转录
✓ 无障碍辅助

🔧 故障排除
-----------

问题：没有转录输出
解决：
  1. 确保音频正在播放
  2. 检查音量是否足够大
  3. 尝试指定具体的音频源

问题：PulseAudio连接失败
解决：
  pulseaudio --start

问题：延迟太高
解决：
  使用 --cuda 加速
  减小 --interval 值

========================================
开始使用吧！播放任何音频，看看实时转录的魔力！
========================================
EOF
