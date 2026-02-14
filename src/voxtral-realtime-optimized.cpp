#include "voxtral.h"
#include <pulse/simple.h>
#include <pulse/error.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <signal.h>
#include <algorithm>

// Configuration
static constexpr int SAMPLE_RATE = 16000;
static constexpr int CHANNELS = 1;
static constexpr int CHUNK_SIZE = 1600;  // 100ms at 16kHz
static constexpr int BUFFER_DURATION_MS = 4000;  // 4 seconds buffer (increased)
static constexpr int BUFFER_SIZE = (SAMPLE_RATE * BUFFER_DURATION_MS) / 1000;
static constexpr int MIN_AUDIO_LENGTH = SAMPLE_RATE * 2;  // 2 seconds minimum (increased)
static constexpr int OVERLAP_SIZE = SAMPLE_RATE / 2;  // 500ms overlap for better context

// VAD (Voice Activity Detection) parameters
static constexpr float VAD_ENERGY_THRESHOLD = 0.015f;  // Increased threshold
static constexpr float VAD_ZCR_THRESHOLD = 0.3f;  // Zero-crossing rate threshold
static constexpr int VAD_MIN_SPEECH_FRAMES = 10;  // Minimum frames to consider as speech

// Audio enhancement parameters
static constexpr float NOISE_GATE_THRESHOLD = 0.005f;
static constexpr float AGC_TARGET_LEVEL = 0.3f;  // Automatic Gain Control target

// Global state
static std::atomic<bool> g_running(true);
static std::mutex g_audio_mutex;
static std::condition_variable g_audio_cv;
static std::deque<float> g_audio_buffer;
static std::atomic<bool> g_new_audio_available(false);

// Performance statistics
struct PerformanceStats {
    std::atomic<int> total_transcriptions{0};
    std::atomic<int> successful_transcriptions{0};
    std::atomic<double> avg_inference_time{0.0};
    std::atomic<double> avg_gpu_usage{0.0};
};
static PerformanceStats g_stats;

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cerr << "\nShutting down...\n";
        g_running = false;
        g_audio_cv.notify_all();
    }
}

// Advanced VAD with energy and zero-crossing rate
struct VADResult {
    bool is_speech;
    float energy;
    float zcr;
    float confidence;
};

VADResult detect_voice_activity(const std::vector<float> & audio) {
    if (audio.empty()) {
        return {false, 0.0f, 0.0f, 0.0f};
    }

    // Calculate energy (RMS)
    float sum_squares = 0.0f;
    for (float sample : audio) {
        sum_squares += sample * sample;
    }
    float energy = std::sqrt(sum_squares / audio.size());

    // Calculate zero-crossing rate
    int zero_crossings = 0;
    for (size_t i = 1; i < audio.size(); ++i) {
        if ((audio[i] >= 0 && audio[i-1] < 0) || (audio[i] < 0 && audio[i-1] >= 0)) {
            zero_crossings++;
        }
    }
    float zcr = static_cast<float>(zero_crossings) / audio.size();

    // Determine if speech based on both energy and ZCR
    bool is_speech = (energy > VAD_ENERGY_THRESHOLD) && (zcr < VAD_ZCR_THRESHOLD);

    // Calculate confidence score
    float energy_score = std::min(energy / (VAD_ENERGY_THRESHOLD * 2.0f), 1.0f);
    float zcr_score = 1.0f - std::min(zcr / VAD_ZCR_THRESHOLD, 1.0f);
    float confidence = (energy_score + zcr_score) / 2.0f;

    return {is_speech, energy, zcr, confidence};
}

// Audio preprocessing: noise gate and AGC
void preprocess_audio(std::vector<float> & audio) {
    if (audio.empty()) return;

    // 1. Noise gate - remove very quiet samples
    for (float & sample : audio) {
        if (std::abs(sample) < NOISE_GATE_THRESHOLD) {
            sample *= 0.1f;  // Attenuate instead of complete silence
        }
    }

    // 2. Automatic Gain Control (AGC)
    float max_amplitude = 0.0f;
    for (float sample : audio) {
        max_amplitude = std::max(max_amplitude, std::abs(sample));
    }

    if (max_amplitude > 0.01f) {  // Avoid division by very small numbers
        float gain = AGC_TARGET_LEVEL / max_amplitude;
        // Limit gain to avoid over-amplification
        gain = std::min(gain, 3.0f);

        for (float & sample : audio) {
            sample *= gain;
            // Soft clipping to avoid distortion
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
        }
    }

    // 3. High-pass filter to remove DC offset and low-frequency noise
    static float prev_input = 0.0f;
    static float prev_output = 0.0f;
    const float alpha = 0.95f;  // High-pass filter coefficient

    for (float & sample : audio) {
        float output = alpha * (prev_output + sample - prev_input);
        prev_input = sample;
        prev_output = output;
        sample = output;
    }
}

// Audio capture thread with optimizations
void audio_capture_thread(const std::string & source_name) {
    pa_sample_spec ss;
    ss.format = PA_SAMPLE_S16LE;
    ss.rate = SAMPLE_RATE;
    ss.channels = CHANNELS;

    pa_buffer_attr buffer_attr;
    buffer_attr.maxlength = (uint32_t) -1;
    buffer_attr.tlength = (uint32_t) -1;
    buffer_attr.prebuf = (uint32_t) -1;
    buffer_attr.minreq = (uint32_t) -1;
    buffer_attr.fragsize = CHUNK_SIZE * sizeof(int16_t);  // Optimize fragment size

    int error;
    pa_simple *s = nullptr;

    // Try to open the specified source or default monitor
    if (!source_name.empty()) {
        s = pa_simple_new(nullptr, "voxtral-realtime", PA_STREAM_RECORD,
                         source_name.c_str(), "Audio Capture", &ss,
                         nullptr, &buffer_attr, &error);
    }

    if (!s) {
        s = pa_simple_new(nullptr, "voxtral-realtime", PA_STREAM_RECORD,
                         nullptr, "Audio Capture", &ss,
                         nullptr, &buffer_attr, &error);
    }

    if (!s) {
        std::cerr << "Failed to open PulseAudio: " << pa_strerror(error) << "\n";
        g_running = false;
        return;
    }

    std::cerr << "Audio capture started (optimized)...\n";

    std::vector<int16_t> chunk(CHUNK_SIZE);
    std::vector<float> float_chunk(CHUNK_SIZE);

    while (g_running) {
        // Read audio chunk
        if (pa_simple_read(s, chunk.data(), chunk.size() * sizeof(int16_t), &error) < 0) {
            std::cerr << "PulseAudio read error: " << pa_strerror(error) << "\n";
            break;
        }

        // Convert to float with better precision
        for (size_t i = 0; i < chunk.size(); ++i) {
            float_chunk[i] = static_cast<float>(chunk[i]) / 32768.0f;
        }

        // Add to buffer with lock
        {
            std::lock_guard<std::mutex> lock(g_audio_mutex);
            for (float sample : float_chunk) {
                g_audio_buffer.push_back(sample);
            }

            // Keep buffer size limited
            while (g_audio_buffer.size() > BUFFER_SIZE) {
                g_audio_buffer.pop_front();
            }

            g_new_audio_available = true;
        }

        // Notify transcription thread
        g_audio_cv.notify_one();
    }

    pa_simple_free(s);
    std::cerr << "Audio capture stopped.\n";
}

// Optimized transcription thread with batching and better scheduling
void transcription_thread(voxtral_context * ctx, int interval_ms, bool show_stats) {
    auto last_transcribe = std::chrono::steady_clock::now();
    std::string last_text;
    int consecutive_silence = 0;
    std::vector<float> overlap_buffer;

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_transcribe).count();

        if (elapsed >= interval_ms) {
            // Get audio from buffer
            std::vector<float> audio;
            {
                std::unique_lock<std::mutex> lock(g_audio_mutex);

                // Wait for new audio if buffer is too small
                if (g_audio_buffer.size() < MIN_AUDIO_LENGTH && g_running) {
                    g_audio_cv.wait_for(lock, std::chrono::milliseconds(100));
                }

                if (g_audio_buffer.size() >= MIN_AUDIO_LENGTH) {
                    // Add overlap from previous segment for better context
                    if (!overlap_buffer.empty()) {
                        audio.insert(audio.end(), overlap_buffer.begin(), overlap_buffer.end());
                    }

                    audio.insert(audio.end(), g_audio_buffer.begin(), g_audio_buffer.end());

                    // Save overlap for next iteration
                    if (g_audio_buffer.size() >= OVERLAP_SIZE) {
                        overlap_buffer.assign(
                            g_audio_buffer.end() - OVERLAP_SIZE,
                            g_audio_buffer.end()
                        );
                    }
                }

                g_new_audio_available = false;
            }

            if (!audio.empty()) {
                // Advanced VAD
                VADResult vad = detect_voice_activity(audio);

                if (vad.is_speech && vad.confidence > 0.3f) {
                    consecutive_silence = 0;

                    // Preprocess audio for better quality
                    preprocess_audio(audio);

                    // Measure inference time
                    auto inference_start = std::chrono::steady_clock::now();

                    // Transcribe
                    voxtral_result result;
                    bool success = voxtral_transcribe_audio(*ctx, audio, 256, result);

                    auto inference_end = std::chrono::steady_clock::now();
                    double inference_time = std::chrono::duration<double, std::milli>(
                        inference_end - inference_start).count();

                    // Update statistics
                    g_stats.total_transcriptions++;
                    if (success && !result.text.empty()) {
                        g_stats.successful_transcriptions++;

                        // Update average inference time
                        double prev_avg = g_stats.avg_inference_time.load();
                        double new_avg = (prev_avg * (g_stats.successful_transcriptions - 1) + inference_time)
                                       / g_stats.successful_transcriptions;
                        g_stats.avg_inference_time = new_avg;
                    }

                    if (success && !result.text.empty() && result.text != last_text) {
                        // Clear line and print new transcription with confidence
                        std::cout << "\r\033[K" << "🎤 [" << std::fixed << std::setprecision(1)
                                  << (vad.confidence * 100) << "%] " << result.text << std::flush;

                        if (show_stats) {
                            std::cerr << " [" << std::fixed << std::setprecision(0)
                                     << inference_time << "ms]";
                        }

                        last_text = result.text;
                    }
                } else {
                    consecutive_silence++;

                    // Clear display after prolonged silence
                    if (consecutive_silence > 5 && !last_text.empty()) {
                        std::cout << "\r\033[K" << "🎤 [Waiting for speech...]" << std::flush;
                        last_text.clear();
                    }
                }
            }

            last_transcribe = now;
        }

        // Adaptive sleep based on audio availability
        if (g_new_audio_available) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void print_usage(const char * argv0) {
    std::cout
        << "Real-time Audio Transcription Client (Optimized)\n"
        << "=================================================\n\n"
        << "Usage: " << argv0 << " --model path.gguf [options]\n\n"
        << "Options:\n"
        << "  --model PATH          GGUF model path (required)\n"
        << "  --source NAME         PulseAudio source name (default: system monitor)\n"
        << "  --interval MS         Transcription interval in milliseconds (default: 1500)\n"
        << "  --threads N           CPU threads (default: 8)\n"
        << "  --cuda                Use CUDA backend\n"
        << "  --metal               Use Metal backend\n"
        << "  --show-stats          Show performance statistics\n"
        << "  --list-sources        List available PulseAudio sources and exit\n"
        << "  -h, --help            Show this help\n\n"
        << "Optimizations:\n"
        << "  • Advanced VAD (Voice Activity Detection)\n"
        << "  • Audio preprocessing (noise gate, AGC, high-pass filter)\n"
        << "  • Overlapping segments for better context\n"
        << "  • Adaptive buffering and scheduling\n"
        << "  • Performance monitoring\n\n";
}

void list_audio_sources() {
    std::cout << "Available PulseAudio sources:\n";
    std::cout << "=============================\n\n";
    int ret = system("pactl list sources | grep -E '(Name:|Description:)'");
    (void)ret;
    std::cout << "\nUse the 'Name:' value with --source option\n";
}

void print_statistics() {
    std::cout << "\n\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Performance Statistics\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Total transcriptions:      " << g_stats.total_transcriptions << "\n";
    std::cout << "  Successful transcriptions: " << g_stats.successful_transcriptions << "\n";

    if (g_stats.successful_transcriptions > 0) {
        float success_rate = (float)g_stats.successful_transcriptions / g_stats.total_transcriptions * 100;
        std::cout << "  Success rate:              " << std::fixed << std::setprecision(1)
                  << success_rate << "%\n";
        std::cout << "  Avg inference time:        " << std::fixed << std::setprecision(0)
                  << g_stats.avg_inference_time << " ms\n";
    }
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string source_name;
    int interval_ms = 1500;  // Reduced default interval for faster response
    int threads = 8;  // Increased default threads
    bool use_cuda = false;
    bool use_metal = false;
    bool list_sources = false;
    bool show_stats = false;

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
        } else if (arg == "--show-stats") {
            show_stats = true;
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
    std::cerr << "Loading model (optimized): " << model_path << "\n";

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

    // Initialize context with optimized parameters
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
    std::cerr << "Real-time Transcription Started (Optimized)\n";
    std::cerr << "========================================\n";
    std::cerr << "Transcription interval: " << interval_ms << "ms\n";
    std::cerr << "CPU threads: " << threads << "\n";
    std::cerr << "Backend: " << (use_cuda ? "CUDA" : (use_metal ? "Metal" : "CPU")) << "\n";
    std::cerr << "Optimizations: VAD, AGC, Noise Gate, Overlap\n";
    std::cerr << "Press Ctrl+C to stop\n";
    std::cerr << "========================================\n\n";

    // Start threads
    std::thread capture_thread(audio_capture_thread, source_name);
    std::thread transcribe_thread(transcription_thread, ctx, interval_ms, show_stats);

    // Wait for threads
    capture_thread.join();
    transcribe_thread.join();

    std::cout << "\n";

    // Print statistics if requested
    if (show_stats) {
        print_statistics();
    }

    // Cleanup
    voxtral_free(ctx);
    voxtral_model_free(model);

    return 0;
}
