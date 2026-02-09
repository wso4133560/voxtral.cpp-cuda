#include "voxtral.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct cli_params {
    std::string model;
    std::string audio;
    std::string prompt;
    std::string dump_logits;
    std::string dump_logits_bin;
    std::string dump_tokens;
    std::string output_text;
    int32_t threads = 0;
    uint32_t seed = 0;
    int32_t max_tokens = 256;
    voxtral_log_level log_level = voxtral_log_level::info;
};

void print_usage(const char * argv0) {
    std::cout
        << "usage: " << argv0 << " --model path.gguf --audio file.wav [options]\n"
        << "\n"
        << "options:\n"
        << "  --model PATH          GGUF model path\n"
        << "  --audio PATH          input wav (mono or stereo PCM16/float32)\n"
        << "  --threads N           cpu threads\n"
        << "  --seed N              rng seed (reserved for sampling)\n"
        << "  --prompt TEXT         prompt text (compatibility, currently ignored for realtime mode)\n"
        << "  --n-tokens N          max decode tokens (alias of --max-len)\n"
        << "  --max-len N           max decode tokens\n"
        << "  --verbose             equivalent to --log-level debug\n"
        << "  --log-level LEVEL     error|warn|info|debug\n"
        << "  --dump-logits PATH    write step-0 logits (first 32 values) as text\n"
        << "  --dump-logits-bin P   write full step-0 logits as float32 raw bytes\n"
        << "  --dump-tokens PATH    write generated token ids as a single line\n"
        << "  --output-text PATH    write decoded text to file (still prints to stdout)\n"
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

bool parse_u32(const std::string & s, uint32_t & out) {
    char * end = nullptr;
    const unsigned long v = std::strtoul(s.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

bool parse_level(const std::string & s, voxtral_log_level & out) {
    if (s == "error") {
        out = voxtral_log_level::error;
        return true;
    }
    if (s == "warn") {
        out = voxtral_log_level::warn;
        return true;
    }
    if (s == "info") {
        out = voxtral_log_level::info;
        return true;
    }
    if (s == "debug") {
        out = voxtral_log_level::debug;
        return true;
    }
    return false;
}

bool parse_args(int argc, char ** argv, cli_params & p) {
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
            p.audio = v;
        } else if (a == "--threads") {
            const char * v = need_value("--threads");
            if (!v || !parse_i32(v, p.threads)) {
                std::cerr << "invalid --threads\n";
                return false;
            }
        } else if (a == "--seed") {
            const char * v = need_value("--seed");
            if (!v || !parse_u32(v, p.seed)) {
                std::cerr << "invalid --seed\n";
                return false;
            }
        } else if (a == "--prompt") {
            const char * v = need_value("--prompt");
            if (!v) {
                return false;
            }
            p.prompt = v;
        } else if (a == "--n-tokens" || a == "--max-len") {
            const char * v = need_value(a.c_str());
            if (!v || !parse_i32(v, p.max_tokens)) {
                std::cerr << "invalid " << a << "\n";
                return false;
            }
        } else if (a == "--verbose") {
            p.log_level = voxtral_log_level::debug;
        } else if (a == "--log-level") {
            const char * v = need_value("--log-level");
            if (!v || !parse_level(v, p.log_level)) {
                std::cerr << "invalid --log-level\n";
                return false;
            }
        } else if (a == "--dump-logits") {
            const char * v = need_value("--dump-logits");
            if (!v) {
                return false;
            }
            p.dump_logits = v;
        } else if (a == "--dump-logits-bin") {
            const char * v = need_value("--dump-logits-bin");
            if (!v) {
                return false;
            }
            p.dump_logits_bin = v;
        } else if (a == "--dump-tokens") {
            const char * v = need_value("--dump-tokens");
            if (!v) {
                return false;
            }
            p.dump_tokens = v;
        } else if (a == "--output-text") {
            const char * v = need_value("--output-text");
            if (!v) {
                return false;
            }
            p.output_text = v;
        } else {
            std::cerr << "unknown option: " << a << "\n";
            return false;
        }
    }

    if (p.model.empty()) {
        std::cerr << "--model is required\n";
        return false;
    }

    if (p.audio.empty()) {
        std::cerr << "--audio is required\n";
        return false;
    }

    if (p.max_tokens <= 0) {
        p.max_tokens = 1;
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    cli_params p;
    if (!parse_args(argc, argv, p)) {
        print_usage(argv[0]);
        return 1;
    }

    voxtral_log_callback logger = [level = p.log_level](voxtral_log_level msg_level, const std::string & msg) {
        if (static_cast<int>(msg_level) > static_cast<int>(level)) {
            return;
        }

        const char * tag = "I";
        if (msg_level == voxtral_log_level::error) {
            tag = "E";
        } else if (msg_level == voxtral_log_level::warn) {
            tag = "W";
        } else if (msg_level == voxtral_log_level::debug) {
            tag = "D";
        }

        std::cerr << "voxtral_" << tag << ": " << msg << "\n";
    };

    voxtral_model * model = voxtral_model_load_from_file(p.model, logger);
    if (!model) {
        return 2;
    }

    voxtral_context_params ctx_p;
    ctx_p.n_threads = p.threads;
    // ctx_p.seed = p.seed;
    ctx_p.log_level = p.log_level;
    ctx_p.logger = logger;

    voxtral_context * ctx = voxtral_init_from_model(model, ctx_p);
    if (!ctx) {
        voxtral_model_free(model);
        return 3;
    }

    voxtral_result result;
    if (!voxtral_transcribe_file(*ctx, p.audio, p.max_tokens, result)) {
        voxtral_free(ctx);
        voxtral_model_free(model);
        return 4;
    }

    const std::string printed_text = result.text.empty() ? std::string("[no-transcript]") : result.text;
    std::cout << printed_text << "\n";

    if (!result.tokens.empty()) {
        std::ostringstream os;
        os << "[tokens]";
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            os << (i == 0 ? " " : " ") << result.tokens[i];
        }
        std::cout << os.str() << "\n";
    } else {
        std::cout << "[no-transcript]\n";
    }

    if (!p.output_text.empty()) {
        std::ofstream fout(p.output_text);
        if (!fout) {
            std::cerr << "failed to open --output-text output\n";
        } else {
            fout << printed_text << "\n";
        }
    }

    if (!p.dump_logits.empty()) {
        std::ofstream fout(p.dump_logits);
        if (!fout) {
            std::cerr << "failed to open --dump-logits output\n";
        } else {
            const size_t n = std::min<size_t>(32, result.first_step_logits.size());
            for (size_t i = 0; i < n; ++i) {
                fout << std::setprecision(9) << result.first_step_logits[i];
                if (i + 1 < n) {
                    fout << "\n";
                }
            }
        }
    }

    if (!p.dump_logits_bin.empty()) {
        std::ofstream fout(p.dump_logits_bin, std::ios::binary);
        if (!fout) {
            std::cerr << "failed to open --dump-logits-bin output\n";
        } else if (!result.first_step_logits.empty()) {
            fout.write(
                reinterpret_cast<const char *>(result.first_step_logits.data()),
                static_cast<std::streamsize>(result.first_step_logits.size() * sizeof(float)));
        }
    }

    if (!p.dump_tokens.empty()) {
        std::ofstream fout(p.dump_tokens);
        if (!fout) {
            std::cerr << "failed to open --dump-tokens output\n";
        } else {
            for (size_t i = 0; i < result.tokens.size(); ++i) {
                if (i > 0) {
                    fout << ' ';
                }
                fout << result.tokens[i];
            }
            fout << "\n";
        }
    }

    voxtral_free(ctx);
    voxtral_model_free(model);
    return 0;
}
