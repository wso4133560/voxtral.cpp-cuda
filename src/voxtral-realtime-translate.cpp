#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <netdb.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

struct options {
    std::string ollama_model = "translategemma:4b-it-q4_K_M";
    std::string ollama_host = "127.0.0.1";
    int         ollama_port = 11434;
    bool        show_original = false;
    std::vector<std::string> child_args;
};

struct transcript_line {
    bool        is_transcript = false;
    bool        is_waiting = false;
    std::string prefix;
    std::string text;
    std::string raw;
};

std::atomic<pid_t> g_child_pid{-1};

void signal_handler(int sig) {
    const pid_t pid = g_child_pid.load();
    if (pid > 0) {
        kill(pid, sig);
    }
}

void print_usage(const char * argv0) {
    std::cout
        << "Real-time Chinese Translation Wrapper\n"
        << "=====================================\n\n"
        << "Usage: " << argv0 << " [translate options] --model path.gguf [voxtral options]\n\n"
        << "Translate options:\n"
        << "  --ollama-model NAME    Ollama model name (default: translategemma:4b-it-q4_K_M)\n"
        << "  --ollama-host HOST:PORT\n"
        << "                         Ollama API endpoint (default: 127.0.0.1:11434)\n"
        << "  --show-original        Show original English text after Chinese\n"
        << "  -h, --help             Show this help\n\n"
        << "All other arguments are forwarded to voxtral-realtime-opt.\n"
        << "Example:\n"
        << "  " << argv0 << " --model models/voxtral/Q4_0.gguf --cuda --show-original\n";
}

bool parse_host_port(const std::string & value, std::string & host, int & port) {
    std::string s = value;
    const std::string http = "http://";
    const std::string https = "https://";
    if (s.rfind(http, 0) == 0) {
        s.erase(0, http.size());
    } else if (s.rfind(https, 0) == 0) {
        s.erase(0, https.size());
    }

    const auto slash = s.find('/');
    if (slash != std::string::npos) {
        s.erase(slash);
    }

    const auto colon = s.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon == s.size() - 1) {
        return false;
    }

    host = s.substr(0, colon);
    port = std::atoi(s.substr(colon + 1).c_str());
    return !host.empty() && port > 0;
}

bool parse_args(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        }
        if (arg == "--ollama-model" && i + 1 < argc) {
            opts.ollama_model = argv[++i];
            continue;
        }
        if (arg == "--ollama-host" && i + 1 < argc) {
            if (!parse_host_port(argv[++i], opts.ollama_host, opts.ollama_port)) {
                std::cerr << "Invalid --ollama-host, expected HOST:PORT\n";
                return false;
            }
            continue;
        }
        if (arg == "--show-original") {
            opts.show_original = true;
            continue;
        }
        opts.child_args.push_back(arg);
    }

    if (opts.child_args.empty()) {
        print_usage(argv[0]);
        return false;
    }

    return true;
}

std::string get_self_dir() {
    std::array<char, 4096> buf{};
    const ssize_t n = readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    if (n <= 0) {
        return ".";
    }
    buf[(size_t) n] = '\0';
    std::string path(buf.data());
    const auto slash = path.rfind('/');
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string json_escape(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 32);
    for (unsigned char c : text) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char tmp[8];
                    std::snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                    out += tmp;
                } else {
                    out.push_back((char) c);
                }
        }
    }
    return out;
}

bool extract_json_string_field(const std::string & body, const std::string & key, std::string & value) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = body.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return false;
    }
    pos = body.find('"', pos + 1);
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;

    std::string out;
    bool escape = false;
    for (; pos < body.size(); ++pos) {
        const char c = body[pos];
        if (escape) {
            switch (c) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: out.push_back(c); break;
            }
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            value = out;
            return true;
        }
        out.push_back(c);
    }

    return false;
}

bool http_post_local(const std::string & host, int port, const std::string & body, std::string & response) {
    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo * result = nullptr;
    const std::string port_str = std::to_string(port);
    const int rc = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
    if (rc != 0) {
        return false;
    }

    int fd = -1;
    for (struct addrinfo * rp = result; rp != nullptr; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd == -1) {
            continue;
        }
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(result);

    if (fd == -1) {
        return false;
    }

    std::string req;
    req += "POST /api/generate HTTP/1.1\r\n";
    req += "Host: " + host + ":" + std::to_string(port) + "\r\n";
    req += "Content-Type: application/json\r\n";
    req += "Connection: close\r\n";
    req += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
    req += body;

    size_t sent = 0;
    while (sent < req.size()) {
        const ssize_t n = send(fd, req.data() + sent, req.size() - sent, 0);
        if (n <= 0) {
            close(fd);
            return false;
        }
        sent += (size_t) n;
    }

    std::array<char, 8192> buf{};
    response.clear();
    while (true) {
        const ssize_t n = recv(fd, buf.data(), buf.size(), 0);
        if (n == 0) {
            break;
        }
        if (n < 0) {
            close(fd);
            return false;
        }
        response.append(buf.data(), (size_t) n);
    }

    close(fd);
    return true;
}

bool translate_text(const options & opts, const std::string & english, std::string & chinese) {
    const std::string prompt =
        "Translate the following English speech transcript into natural Simplified Chinese. "
        "Keep names, numbers, and punctuation accurate. Return only the Chinese translation.\n\n"
        "English:\n" + english + "\n";

    const std::string body =
        "{\"model\":\"" + json_escape(opts.ollama_model) +
        "\",\"prompt\":\"" + json_escape(prompt) +
        "\",\"stream\":false,\"options\":{\"temperature\":0}}";

    std::string http_response;
    if (!http_post_local(opts.ollama_host, opts.ollama_port, body, http_response)) {
        return false;
    }

    const auto header_end = http_response.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return false;
    }

    const std::string header = http_response.substr(0, header_end);
    if (header.find("200 OK") == std::string::npos) {
        return false;
    }

    const std::string payload = http_response.substr(header_end + 4);
    if (!extract_json_string_field(payload, "response", chinese)) {
        return false;
    }

    while (!chinese.empty() && (chinese.back() == '\n' || chinese.back() == '\r' || chinese.back() == ' ')) {
        chinese.pop_back();
    }
    return !chinese.empty();
}

std::string strip_ansi(const std::string & input) {
    std::string out;
    out.reserve(input.size());
    enum class ansi_state {
        normal,
        esc,
        csi,
    };
    ansi_state state = ansi_state::normal;

    for (size_t i = 0; i < input.size(); ++i) {
        const char c = input[i];
        if (state == ansi_state::normal) {
            if (c == '\x1b') {
                state = ansi_state::esc;
            } else if (c != '\r') {
                out.push_back(c);
            }
            continue;
        }

        if (state == ansi_state::esc) {
            if (c == '[') {
                state = ansi_state::csi;
            } else if (c >= '@' && c <= '~') {
                state = ansi_state::normal;
            }
            continue;
        }

        if (c >= '@' && c <= '~') {
            state = ansi_state::normal;
        }
    }
    return out;
}

transcript_line parse_line(const std::string & raw_line) {
    transcript_line line;
    line.raw = strip_ansi(raw_line);

    while (!line.raw.empty() && (line.raw.front() == ' ' || line.raw.front() == '\n')) {
        line.raw.erase(line.raw.begin());
    }
    while (!line.raw.empty() && (line.raw.back() == ' ' || line.raw.back() == '\n')) {
        line.raw.pop_back();
    }

    if (line.raw.rfind("\xF0\x9F\x8E\xA4 ", 0) != 0) {
        return line;
    }

    line.is_transcript = true;
    if (line.raw.find("Waiting for speech") != std::string::npos) {
        line.is_waiting = true;
        return line;
    }

    const size_t bracket = line.raw.find("] ");
    if (bracket == std::string::npos) {
        line.is_transcript = false;
        return line;
    }

    line.prefix = line.raw.substr(0, bracket + 1);
    line.text = line.raw.substr(bracket + 2);
    return line;
}

void print_translated_line(const transcript_line & line, const std::string & chinese, bool show_original) {
    std::cout << "\r\033[K";
    if (line.is_waiting) {
        std::cout << "🌏 [等待语音输入...]" << std::flush;
        return;
    }

    std::cout << "🌏 ";
    if (!line.prefix.empty()) {
        std::cout << line.prefix.substr(std::string("\xF0\x9F\x8E\xA4 ").size()) << ' ';
    }
    std::cout << chinese;
    if (show_original) {
        std::cout << " | EN: " << line.text;
    }
    std::cout << std::flush;
}

void print_passthrough_line(const std::string & line) {
    if (line.empty()) {
        return;
    }
    std::cout << "\n" << line << std::flush;
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_args(argc, argv, opts)) {
        return opts.child_args.empty() ? 0 : 1;
    }

    const std::string child_path = get_self_dir() + "/voxtral-realtime-opt";
    if (access(child_path.c_str(), X_OK) != 0) {
        std::cerr << "Failed to find executable: " << child_path << "\n";
        return 2;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        std::cerr << "pipe() failed: " << std::strerror(errno) << "\n";
        return 3;
    }

    std::vector<std::string> child_storage;
    child_storage.reserve(opts.child_args.size() + 1);
    child_storage.push_back(child_path);
    for (const auto & arg : opts.child_args) {
        child_storage.push_back(arg);
    }

    std::vector<char *> child_argv;
    child_argv.reserve(child_storage.size() + 1);
    for (auto & arg : child_storage) {
        child_argv.push_back(arg.data());
    }
    child_argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        std::cerr << "fork() failed: " << std::strerror(errno) << "\n";
        return 4;
    }

    if (pid == 0) {
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
        execv(child_path.c_str(), child_argv.data());
        std::perror("execv");
        _exit(127);
    }

    g_child_pid = pid;
    close(pipefd[1]);

    std::unordered_map<std::string, std::string> cache;
    std::string buffer;
    std::string last_english;

    std::array<char, 1024> chunk{};
    while (true) {
        const ssize_t n = read(pipefd[0], chunk.data(), chunk.size());
        if (n == 0) {
            break;
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }

        for (ssize_t i = 0; i < n; ++i) {
            const char c = chunk[(size_t) i];
            if (c != '\r' && c != '\n') {
                buffer.push_back(c);
                continue;
            }

            const transcript_line line = parse_line(buffer);
            if (line.is_transcript) {
                if (line.is_waiting) {
                    print_translated_line(line, "", opts.show_original);
                } else if (!line.text.empty() && line.text != last_english) {
                    auto it = cache.find(line.text);
                    std::string chinese;
                    if (it != cache.end()) {
                        chinese = it->second;
                    } else if (translate_text(opts, line.text, chinese)) {
                        cache.emplace(line.text, chinese);
                    } else {
                        chinese = "[Ollama translation failed] " + line.text;
                    }
                    print_translated_line(line, chinese, opts.show_original);
                    last_english = line.text;
                }
            } else {
                const std::string cleaned = strip_ansi(buffer);
                if (!cleaned.empty()) {
                    print_passthrough_line(cleaned);
                }
            }
            buffer.clear();
        }
    }

    close(pipefd[0]);

    if (!buffer.empty()) {
        const std::string cleaned = strip_ansi(buffer);
        if (!cleaned.empty()) {
            print_passthrough_line(cleaned);
        }
    }

    int status = 0;
    waitpid(pid, &status, 0);
    g_child_pid = -1;
    std::cout << "\n";

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 5;
}
