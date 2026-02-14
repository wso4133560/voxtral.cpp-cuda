#include "voxtral.h"
#include <pulse/simple.h>
#include <pulse/error.h>
#include <iostream>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <signal.h>

// Configuration
static constexpr int SAMPLE_RATE = 16000;
static constexpr int CHANNELS = 1;
static constexpr int CHUNK_SIZE = 1600;  // 100ms at 16kHz
static constexpr int BUFFER_DURATION_MS = 3000;  // 3 seconds buffer
static constexpr int BUFFER_SIZE = (SAMPLE_RATE * BUFFER_DURATION_MS) / 1000;
static constexpr int MIN_AUDIO_LENGTH = SAMPLE_RATE * 1;  // 1 second minimum

// Global state
static std::atomic<bool> g_running(true);
static std::mutex g_audio_mutex;
static std::deque<float> g_audio_buffer;

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cerr << "\nShutting down...\n";
        g_running = false;
    }
}

// Audio capture thread
void audio_capture_thread(const std::string & source_name) {
    pa_sample_spec ss;
    ss.format = PA_SAMPLE_S16LE;
    ss.rate = SAMPLE_RATE;
    ss.channels = CHANNELS;

    int error;
    pa_simple *s = nullptr;

    // Try to open the specified source or default monitor
    if (!source_name.empty()) {
        s = pa_simple_new(nullptr, "voxtral-realtime", PA_STREAM_RECORD,
                         source_name.c_str(), "Audio Capture", &ss,
                         nullptr, nullptr, &error);
    }

    if (!s) {
        // Fallback to default monitor
        s = pa_simple_new(nullptr, "voxtral-realtime", PA_STREAM_RECORD,
                         nullptr, "Audio Capture", &ss,
                         nullptr, nullptr, &error);
    }

    if (!s) {
        std::cerr << "Failed to open PulseAudio: " << pa_strerror(error) << "\n";
        g_running = false;
        return;
    }

    std::cerr << "Audio capture started...\n";

    std::vector<int16_t> chunk(CHUNK_SIZE);

    while (g_running) {
        // Read audio chunk
        if (pa_simple_read(s, chunk.data(), chunk.size() * sizeof(int16_t), &error) < 0) {
            std::cerr << "PulseAudio read error: " << pa_strerror(error) << "\n";
            break;
        }

        // Convert to float and add to buffer
        std::lock_guard<std::mutex> lock(g_audio_mutex);
        for (int16_t sample : chunk) {
            float f = static_cast<float>(sample) / 32768.0f;
            g_audio_buffer.push_back(f);
        }

        // Keep buffer size limited
        while (g_audio_buffer.size() > BUFFER_SIZE) {
            g_audio_buffer.pop_front();
        }
    }

    pa_simple_free(s);
    std::cerr << "Audio capture stopped.\n";
}

// Check if audio has significant energy (not silence)
bool has_audio_energy(const std::vector<float> & audio, float threshold = 0.01f) {
    if (audio.empty()) return false;

    float sum = 0.0f;
    for (float sample : audio) {
        sum += sample * sample;
    }
    float rms = std::sqrt(sum / audio.size());
    return rms > threshold;
}

// Transcription thread
void transcription_thread(voxtral_context * ctx, int interval_ms) {
    auto last_transcribe = std::chrono::steady_clock::now();
    std::string last_text;

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_transcribe).count();

        if (elapsed >= interval_ms) {
            // Get audio from buffer
            std::vector<float> audio;
            {
                std::lock_guard<std::mutex> lock(g_audio_mutex);
                if (g_audio_buffer.size() >= MIN_AUDIO_LENGTH) {
                    audio.assign(g_audio_buffer.begin(), g_audio_buffer.end());
                }
            }

            if (!audio.empty() && has_audio_energy(audio)) {
                // Transcribe
                voxtral_result result;
                if (voxtral_transcribe_audio(*ctx, audio, 256, result)) {
                    if (!result.text.empty() && result.text != last_text) {
                        // Clear line and print new transcription
                        std::cout << "\r\033[K" << "🎤 " << result.text << std::flush;
                        last_text = result.text;
                    }
                }
            }

            last_transcribe = now;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void print_usage(const char * argv0) {
    std::cout
        << "Real-time Audio Transcription Client\n"
        << "=====================================\n\n"
        << "Usage: " << argv0 << " --model path.gguf [options]\n\n"
        << "Options:\n"
        << "  --model PATH          GGUF model path (required)\n"
        << "  --source NAME         PulseAudio source name (default: system monitor)\n"
        << "  --interval MS         Transcription interval in milliseconds (default: 2000)\n"
        << "  --threads N           CPU threads (default: 4)\n"
        << "  --cuda                Use CUDA backend\n"
        << "  --metal               Use Metal backend\n"
        << "  --list-sources        List available PulseAudio sources and exit\n"
        << "  -h, --help            Show this help\n\n"
        << "Examples:\n"
        << "  # Use default system audio monitor\n"
        << "  " << argv0 << " --model models/voxtral/Q4_0.gguf --cuda\n\n"
        << "  # Use specific audio source\n"
        << "  " << argv0 << " --model models/voxtral/Q4_0.gguf --source alsa_output.pci-0000_00_1f.3.analog-stereo.monitor\n\n"
        << "  # List available sources\n"
        << "  " << argv0 << " --list-sources\n\n";
}

void list_audio_sources() {
    std::cout << "Available PulseAudio sources:\n";
    std::cout << "=============================\n\n";
    int ret = system("pactl list sources | grep -E '(Name:|Description:)'");
    (void)ret;  // Suppress unused result warning
    std::cout << "\nUse the 'Name:' value with --source option\n";
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string source_name;
    int interval_ms = 2000;
    int threads = 4;
    bool use_cuda = false;
    bool use_metal = false;
    bool list_sources = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--list-sources") {
            list_sources = true;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            source_name = argv[++i];
        } else if (arg == "--interval" && i + 1 < argc) {
            interval_ms = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--metal") {
            use_metal = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (list_sources) {
        list_audio_sources();
        return 0;
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Load model
    std::cerr << "Loading model: " << model_path << "\n";

    auto logger = [](voxtral_log_level level, const std::string & msg) {
        if (level <= voxtral_log_level::info) {
            std::cerr << "voxtral: " << msg << "\n";
        }
    };

    voxtral_model * model = voxtral_model_load_from_file_ex(
        model_path, logger, use_metal, use_cuda);

    if (!model) {
        std::cerr << "Failed to load model\n";
        return 2;
    }

    // Initialize context
    voxtral_context_params ctx_params;
    ctx_params.n_threads = threads;
    ctx_params.log_level = voxtral_log_level::warn;
    ctx_params.logger = logger;
    ctx_params.use_metal = use_metal;
    ctx_params.use_cuda = use_cuda;

    voxtral_context * ctx = voxtral_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to initialize context\n";
        voxtral_model_free(model);
        return 3;
    }

    std::cerr << "\n";
    std::cerr << "========================================\n";
    std::cerr << "Real-time Transcription Started\n";
    std::cerr << "========================================\n";
    std::cerr << "Transcription interval: " << interval_ms << "ms\n";
    std::cerr << "Press Ctrl+C to stop\n";
    std::cerr << "========================================\n\n";

    // Start threads
    std::thread capture_thread(audio_capture_thread, source_name);
    std::thread transcribe_thread(transcription_thread, ctx, interval_ms);

    // Wait for threads
    capture_thread.join();
    transcribe_thread.join();

    std::cout << "\n";

    // Cleanup
    voxtral_free(ctx);
    voxtral_model_free(model);

    return 0;
}
