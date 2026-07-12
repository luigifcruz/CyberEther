#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "jetstream/run.hh"

namespace {

#if defined(_WIN32)
int FileDescriptor(FILE* stream) { return _fileno(stream); }
int Duplicate(int descriptor) { return _dup(descriptor); }
int Redirect(int source, int destination) { return _dup2(source, destination); }
void Close(int descriptor) { (void)_close(descriptor); }
#else
int FileDescriptor(FILE* stream) { return fileno(stream); }
int Duplicate(int descriptor) { return dup(descriptor); }
int Redirect(int source, int destination) { return dup2(source, destination); }
void Close(int descriptor) { (void)close(descriptor); }
#endif

class StreamCapture {
 public:
    explicit StreamCapture(FILE* stream) : stream_(stream), descriptor_(FileDescriptor(stream)) {
        file_ = std::tmpfile();
        savedDescriptor_ = Duplicate(descriptor_);
        REQUIRE(file_ != nullptr);
        REQUIRE(savedDescriptor_ >= 0);
        std::fflush(stream_);
        REQUIRE(Redirect(FileDescriptor(file_), descriptor_) >= 0);
    }

    ~StreamCapture() {
        if (!finished_) {
            (void)finish();
        }
    }

    std::string finish() {
        std::fflush(stream_);
        std::rewind(file_);

        std::string output;
        char buffer[4096];
        while (const size_t size = std::fread(buffer, 1, sizeof(buffer), file_)) {
            output.append(buffer, size);
        }

        (void)Redirect(savedDescriptor_, descriptor_);
        Close(savedDescriptor_);
        std::fclose(file_);
        finished_ = true;
        return output;
    }

 private:
    FILE* stream_;
    FILE* file_ = nullptr;
    int descriptor_;
    int savedDescriptor_ = -1;
    bool finished_ = false;
};

struct InvocationResult {
    int code;
    std::string out;
    std::string err;
};

InvocationResult Invoke(std::initializer_list<const char*> arguments) {
    std::vector<std::string> values = {"cyberether"};
    for (const char* argument : arguments) {
        values.emplace_back(argument);
    }

    std::vector<char*> argv;
    argv.reserve(values.size());
    for (auto& value : values) {
        argv.push_back(value.data());
    }

    StreamCapture out(stdout);
    StreamCapture err(stderr);
    const int code = Jetstream::Run(static_cast<int>(argv.size()), argv.data());
    return {code, out.finish(), err.finish()};
}

void Expect(const char* label,
            std::initializer_list<const char*> arguments,
            int code,
            std::initializer_list<const char*> out = {},
            std::initializer_list<const char*> err = {},
            std::initializer_list<const char*> absentOut = {},
            std::initializer_list<const char*> absentErr = {}) {
    INFO("CLI case: " << label);
    const InvocationResult result = Invoke(arguments);
    CHECK(result.code == code);
    for (const char* value : out) {
        CHECK(result.out.find(value) != std::string::npos);
    }
    for (const char* value : err) {
        CHECK(result.err.find(value) != std::string::npos);
    }
    for (const char* value : absentOut) {
        CHECK(result.out.find(value) == std::string::npos);
    }
    for (const char* value : absentErr) {
        CHECK(result.err.find(value) == std::string::npos);
    }
}

void ConfigureSettingsSandbox() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() /
                      ("cyberether-cli-" + std::to_string(nonce));
    std::filesystem::create_directories(root);

#if defined(_WIN32)
    if (_wputenv_s(L"APPDATA", root.wstring().c_str()) != 0) {
        throw std::runtime_error("failed to configure CLI test settings sandbox");
    }
#elif defined(__APPLE__)
    if (setenv("CFFIXED_USER_HOME", root.string().c_str(), 1) != 0) {
        throw std::runtime_error("failed to configure CLI test settings sandbox");
    }
#else
    if (setenv("HOME", root.string().c_str(), 1) != 0 ||
        setenv("XDG_CONFIG_HOME", root.string().c_str(), 1) != 0) {
        throw std::runtime_error("failed to configure CLI test settings sandbox");
    }
#endif
}

}  // namespace

TEST_CASE("CLI displays contextual help and version", "[cli]") {
    Expect("global help",
           {"--help"},
           0,
           {"Usage:\n", "Commands:\n", "Graphics Options:\n", "Benchmark Options:\n"},
           {},
           {},
           {"Error:"});
    Expect("run help",
           {"run", "--help"},
           0,
           {"run [options] [flowgraph]", "Graphics Options:\n"},
           {},
           {"Benchmark Options:\n"});
    Expect("benchmark help",
           {"benchmark", "--help"},
           0,
           {"benchmark [options] [block]", "Benchmark Options:\n"},
           {},
           {"Graphics Options:\n"});
    Expect("command ordering", {"-v", "run", "--help"}, 0, {"run [options] [flowgraph]"});
    Expect("version", {"--version"}, 0, {"CyberEther v"}, {}, {}, {"Error:"});
}

TEST_CASE("CLI accepts normalized and inline values", "[cli]") {
    Expect("run values",
           {"--renderer=METAL",
            "--device-index=7",
            "--plugin=first.cep",
            "--plugin",
            "second.cep",
            "--size=640X480",
            "--scale=2",
            "--framerate=30",
            "--remote",
            "--broker=https://example.com",
            "--codec=H264",
            "--encoder=AUTO",
            "--auto-join-sessions",
            "--help"},
           0,
           {"Preferred graphics backend (current: metal)",
            "Vulkan and CUDA device index (current: 7)",
            "Viewport size (current: 640x480)",
            "Interface scale factor (current: 2.0)",
            "Target frame rate (current: 30)",
            "Broker URL (current: https://example.com)",
            "Streaming codec (current: h264)",
            "Streaming encoder (current: auto)"});
    Expect("benchmark values",
           {"--format=JSON", "benchmark", "fft", "--plugin=benchmark.cep", "--help"},
           0,
           {"benchmark [options] [block]", "Output format (current: json)"});
    Expect("zero device index",
           {"--device-index", "0", "--help"},
           0,
           {"Vulkan and CUDA device index (current: 0)"});
}

TEST_CASE("CLI rejects invalid syntax and command conflicts", "[cli]") {
    Expect("malformed delimiter value", {"--=value", "--help"}, 2, {}, {"Unknown option: '--=value'."});
    Expect("empty malformed delimiter", {"--=", "--help"}, 2, {}, {"Unknown option: '--='."});
    Expect("missing plugin path", {"--plugin"}, 2, {}, {"Missing value for --plugin"});
    Expect("dash-prefixed flowgraph", {"--", "--flowgraph.yml", "second.yml"}, 2, {}, {"Only one flowgraph"});
    Expect("benchmark delimiter", {"benchmark", "--", "--block", "second"}, 2, {}, {"Only one benchmark block"});
    Expect("multiple flowgraphs", {"one.yml", "two.yml"}, 2, {}, {"Only one flowgraph"});
    Expect("multiple benchmark blocks", {"benchmark", "fft", "am"}, 2, {}, {"Only one benchmark block"});
    Expect("run option with benchmark", {"benchmark", "--headless"}, 2, {}, {"not available for the benchmark command"});
    Expect("device index with benchmark",
           {"benchmark", "--device-index", "7", "--help"},
           0,
           {"Vulkan and CUDA device index (current: 7)"});
    Expect("benchmark option with run", {"run", "--format", "json"}, 2, {}, {"only available for the benchmark command"});
}

TEST_CASE("CLI enforces remote option dependencies", "[cli]") {
    Expect("broker", {"--broker", "https://example.com"}, 2, {}, {"requires --remote"});
    Expect("codec", {"--codec", "h264"}, 2, {}, {"requires --remote"});
    Expect("encoder", {"--encoder", "auto"}, 2, {}, {"requires --remote"});
    Expect("auto join", {"--auto-join-sessions"}, 2, {}, {"requires --remote"});
    Expect("headless value", {"--headless=true"}, 2, {}, {"does not accept a value"});
    Expect("remote value", {"--remote=true"}, 2, {}, {"does not accept a value"});
}

TEST_CASE("CLI validates numeric values", "[cli]") {
    const std::vector<std::pair<const char*, const char*>> errors = {
        {"--size", "Missing value for --size"},
        {"--size=0x480", "Invalid value for --size"},
        {"--size=640x0", "Invalid value for --size"},
        {"--size=640x480x2", "Invalid value for --size"},
        {"--size=2147483648x480", "Invalid value for --size"},
        {"--size=18446744073709551616x480", "Invalid value for --size"},
        {"--scale=0", "Invalid value for --scale"},
        {"--scale=-1", "Invalid value for --scale"},
        {"--scale=nan", "Invalid value for --scale"},
        {"--scale=inf", "Invalid value for --scale"},
        {"--framerate=0", "Invalid value for --framerate"},
        {"--framerate=-1", "Invalid value for --framerate"},
        {"--framerate=18446744073709551616", "Invalid value for --framerate"},
        {"--device-index", "Missing value for --device-index"},
        {"--device-index=-1", "Invalid value for --device-index"},
        {"--device-index=invalid", "Invalid value for --device-index"},
        {"--device-index=18446744073709551616", "Invalid value for --device-index"},
    };
    for (const auto& [argument, message] : errors) {
        Expect(argument, {argument}, 2, {}, {message});
    }
}

TEST_CASE("CLI rejects removed commands and options", "[cli]") {
    Expect("remote command", {"remote"}, 2, {}, {"Unknown command: 'remote'."});
    Expect("device option", {"--device"}, 2, {}, {"Unknown option: '--device'."});
    Expect("device ID option", {"--device-id"}, 2, {}, {"Unknown option: '--device-id'."});
    Expect("endpoint option", {"--endpoint"}, 2, {}, {"Unknown option: '--endpoint'."});
    Expect("auto join option", {"--auto-join"}, 2, {}, {"Unknown option: '--auto-join'."});
}

int main(int argc, char* argv[]) {
    ConfigureSettingsSandbox();
    return Catch::Session().run(argc, argv);
}
