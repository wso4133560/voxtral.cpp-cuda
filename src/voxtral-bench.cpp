#include "voxtral.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct bench_params {
    std::string model;
    std::vector<std::string> audios;
    int32_t threads = 0;
    int32_t max_tokens = 128;
    int32_t warmup = 0;
    bool use_cuda = false;
    bool use_metal = false;
    bool verbose = false;
};

struct bench_result {
    std::string audio_path;
    std::string transcript;
    std::string expected_text;
    int32_t tokens = 0;
    double audio_seconds = 0.0;
    double infer_seconds = 0.0;
    double tokens_per_s = 0.0;
    double rtf = 0.0;
    double wer = 0.0;
};

void print_usage(const char * argv0) {
    std::cout
        << "usage: " << argv0 << " --model path.gguf --audio file.wav [--audio file2.wav ...] [options]\n"
        << "\n"
        << "options:\n"
        << "  --model PATH          GGUF model path\n"
        << "  --audio PATH          input wav, repeatable\n"
        << "  --threads N           cpu threads\n"
        << "  --max-len N           max decode tokens\n"
        << "  --warmup N            warmup runs before measuring (default: 0)\n"
        << "  --cuda                use CUDA backend when available\n"
        << "  --metal               use Metal backend when available\n"
        << "  --verbose             show debug logs\n"
        << "  -h, --help            show this help\n";
}

bool parse_i32(const std::string & s, int32_t & out) {
    char * end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<int32_t>(v);
    return true;
}

bool parse_args(int argc, char ** argv, bench_params & p) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--model") {
            const char * v = need_value("--model");
            if (!v) {
                return false;
            }
            p.model = v;
        } else if (a == "--audio") {
            const char * v = need_value("--audio");
            if (!v) {
                return false;
            }
            p.audios.push_back(v);
        } else if (a == "--threads") {
            const char * v = need_value("--threads");
            if (!v || !parse_i32(v, p.threads)) {
                std::cerr << "invalid --threads\n";
                return false;
            }
        } else if (a == "--max-len") {
            const char * v = need_value("--max-len");
            if (!v || !parse_i32(v, p.max_tokens)) {
                std::cerr << "invalid --max-len\n";
                return false;
            }
        } else if (a == "--warmup") {
            const char * v = need_value("--warmup");
            if (!v || !parse_i32(v, p.warmup)) {
                std::cerr << "invalid --warmup\n";
                return false;
            }
        } else if (a == "--cuda") {
            p.use_cuda = true;
        } else if (a == "--metal") {
            p.use_metal = true;
        } else if (a == "--verbose") {
            p.verbose = true;
        } else {
            std::cerr << "unknown option: " << a << "\n";
            return false;
        }
    }

    if (p.model.empty()) {
        std::cerr << "--model is required\n";
        return false;
    }
    if (p.audios.empty()) {
        std::cerr << "at least one --audio is required\n";
        return false;
    }
    if (p.max_tokens <= 0) {
        p.max_tokens = 1;
    }
    if (p.warmup < 0) {
        p.warmup = 0;
    }
    return true;
}

std::string normalize_text(const std::string & text) {
    std::string out;
    out.reserve(text.size());

    bool last_was_space = true;
    for (unsigned char ch : text) {
        if (std::isalnum(ch) || ch == '\'') {
            out.push_back(static_cast<char>(std::toupper(ch)));
            last_was_space = false;
        } else if (!last_was_space) {
            out.push_back(' ');
            last_was_space = true;
        }
    }

    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

double compute_wer(const std::string & reference, const std::string & hypothesis) {
    std::istringstream ref_in(normalize_text(reference));
    std::istringstream hyp_in(normalize_text(hypothesis));
    std::vector<std::string> ref_words;
    std::vector<std::string> hyp_words;
    for (std::string word; ref_in >> word;) {
        ref_words.push_back(word);
    }
    for (std::string word; hyp_in >> word;) {
        hyp_words.push_back(word);
    }

    if (ref_words.empty()) {
        return 0.0;
    }

    std::vector<std::vector<int>> dist(ref_words.size() + 1, std::vector<int>(hyp_words.size() + 1, 0));
    for (size_t i = 0; i <= ref_words.size(); ++i) {
        dist[i][0] = static_cast<int>(i);
    }
    for (size_t j = 0; j <= hyp_words.size(); ++j) {
        dist[0][j] = static_cast<int>(j);
    }

    for (size_t i = 1; i <= ref_words.size(); ++i) {
        for (size_t j = 1; j <= hyp_words.size(); ++j) {
            if (ref_words[i - 1] == hyp_words[j - 1]) {
                dist[i][j] = dist[i - 1][j - 1];
            } else {
                dist[i][j] = std::min({dist[i - 1][j], dist[i][j - 1], dist[i - 1][j - 1]}) + 1;
            }
        }
    }

    return static_cast<double>(dist[ref_words.size()][hyp_words.size()]) /
           static_cast<double>(ref_words.size());
}

bool read_wav_duration(const std::string & path, double & seconds_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char riff[4];
    char wave[4];
    in.read(riff, 4);
    in.ignore(4);
    in.read(wave, 4);
    if (!in || std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        return false;
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_size = 0;

    while (in && (!sample_rate || !data_size)) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        in.read(chunk_id, 4);
        in.read(reinterpret_cast<char *>(&chunk_size), sizeof(chunk_size));
        if (!in) {
            break;
        }

        const std::string id(chunk_id, 4);
        if (id == "fmt ") {
            in.read(reinterpret_cast<char *>(&audio_format), sizeof(audio_format));
            in.read(reinterpret_cast<char *>(&channels), sizeof(channels));
            in.read(reinterpret_cast<char *>(&sample_rate), sizeof(sample_rate));
            in.ignore(6);
            in.read(reinterpret_cast<char *>(&bits_per_sample), sizeof(bits_per_sample));
            if (chunk_size > 16) {
                in.ignore(chunk_size - 16);
            }
        } else if (id == "data") {
            data_size = chunk_size;
            in.ignore(chunk_size);
        } else {
            in.ignore(chunk_size);
        }

        if (chunk_size % 2 == 1) {
            in.ignore(1);
        }
    }

    if (audio_format == 0 || channels == 0 || sample_rate == 0 || bits_per_sample == 0 || data_size == 0) {
        return false;
    }

    const double bytes_per_sample = static_cast<double>(channels) * static_cast<double>(bits_per_sample / 8);
    if (bytes_per_sample <= 0.0) {
        return false;
    }

    seconds_out = static_cast<double>(data_size) / (static_cast<double>(sample_rate) * bytes_per_sample);
    return true;
}

std::string read_expected_text(const std::string & audio_path) {
    const size_t dot = audio_path.find_last_of('.');
    if (dot == std::string::npos) {
        return {};
    }
    const std::string txt_path = audio_path.substr(0, dot) + ".txt";
    std::ifstream in(txt_path);
    if (!in) {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char ** argv) {
    bench_params p;
    if (!parse_args(argc, argv, p)) {
        print_usage(argv[0]);
        return 1;
    }

    const auto t_load_start = std::chrono::steady_clock::now();
    voxtral_log_callback logger = [verbose = p.verbose](voxtral_log_level level, const std::string & msg) {
        if (!verbose && level > voxtral_log_level::warn) {
            return;
        }
        std::cerr << "voxtral: " << msg << "\n";
    };

    voxtral_model * model = voxtral_model_load_from_file_ex(p.model, logger, p.use_metal, p.use_cuda);
    if (!model) {
        return 2;
    }

    voxtral_context_params ctx_p;
    ctx_p.n_threads = p.threads;
    ctx_p.log_level = p.verbose ? voxtral_log_level::debug : voxtral_log_level::warn;
    ctx_p.logger = logger;
    ctx_p.use_metal = p.use_metal;
    ctx_p.use_cuda = p.use_cuda;

    voxtral_context * ctx = voxtral_init_from_model(model, ctx_p);
    if (!ctx) {
        voxtral_model_free(model);
        return 3;
    }

    if (p.use_cuda && !voxtral_context_uses_cuda(*ctx)) {
        std::cerr << "voxtral-bench: requested --cuda but CUDA backend is unavailable; refusing CPU fallback\n";
        voxtral_free(ctx);
        voxtral_model_free(model);
        return 7;
    }
    const double load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load_start).count();

    for (int32_t i = 0; i < p.warmup; ++i) {
        voxtral_result warmup_result;
        if (!voxtral_transcribe_file(*ctx, p.audios[0], p.max_tokens, warmup_result)) {
            std::cerr << "warmup failed on " << p.audios[0] << "\n";
            voxtral_free(ctx);
            voxtral_model_free(model);
            return 4;
        }
    }

    std::vector<bench_result> results;
    results.reserve(p.audios.size());

    for (const std::string & audio_path : p.audios) {
        bench_result out;
        out.audio_path = audio_path;
        out.expected_text = read_expected_text(audio_path);
        if (!read_wav_duration(audio_path, out.audio_seconds)) {
            std::cerr << "failed to read WAV metadata: " << audio_path << "\n";
            voxtral_free(ctx);
            voxtral_model_free(model);
            return 5;
        }

        voxtral_result transcript;
        const auto t_infer_start = std::chrono::steady_clock::now();
        if (!voxtral_transcribe_file(*ctx, audio_path, p.max_tokens, transcript)) {
            std::cerr << "transcription failed on " << audio_path << "\n";
            voxtral_free(ctx);
            voxtral_model_free(model);
            return 6;
        }
        out.infer_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_infer_start).count();
        out.transcript = transcript.text;
        out.tokens = static_cast<int32_t>(transcript.tokens.size());
        out.tokens_per_s = out.infer_seconds > 0.0 ? static_cast<double>(out.tokens) / out.infer_seconds : 0.0;
        out.rtf = out.audio_seconds > 0.0 ? out.infer_seconds / out.audio_seconds : 0.0;
        out.wer = out.expected_text.empty() ? 0.0 : compute_wer(out.expected_text, out.transcript);
        results.push_back(out);
    }

    std::cout << "model_load_s=" << std::fixed << std::setprecision(3) << load_seconds << "\n";
    std::cout << std::left
              << std::setw(22) << "Sample"
              << std::setw(8)  << "Audio"
              << std::setw(10) << "Infer"
              << std::setw(8)  << "Tokens"
              << std::setw(10) << "Tok/s"
              << std::setw(8)  << "RTF"
              << std::setw(8)  << "WER"
              << "\n";
    std::cout << std::string(74, '-') << "\n";

    double total_audio = 0.0;
    double total_infer = 0.0;
    int32_t total_tokens = 0;
    double total_wer = 0.0;

    for (const bench_result & r : results) {
        std::cout << std::left
                  << std::setw(22) << r.audio_path
                  << std::setw(8)  << std::fixed << std::setprecision(2) << r.audio_seconds
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.infer_seconds
                  << std::setw(8)  << r.tokens
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.tokens_per_s
                  << std::setw(8)  << std::fixed << std::setprecision(2) << r.rtf
                  << std::setw(8)  << std::fixed << std::setprecision(4) << r.wer
                  << "\n";
        total_audio += r.audio_seconds;
        total_infer += r.infer_seconds;
        total_tokens += r.tokens;
        total_wer += r.wer;
    }

    if (!results.empty()) {
        const double avg_wer = total_wer / static_cast<double>(results.size());
        const double agg_tps = total_infer > 0.0 ? static_cast<double>(total_tokens) / total_infer : 0.0;
        const double agg_rtf = total_audio > 0.0 ? total_infer / total_audio : 0.0;
        std::cout << std::string(74, '-') << "\n";
        std::cout << "aggregate_tok_s=" << std::fixed << std::setprecision(3) << agg_tps
                  << " aggregate_rtf=" << agg_rtf
                  << " avg_wer=" << avg_wer
                  << " avg_accuracy=" << (1.0 - avg_wer)
                  << "\n";
    }

    voxtral_free(ctx);
    voxtral_model_free(model);
    return 0;
}
