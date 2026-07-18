#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/run.hh"
#include "jetstream/settings.hh"

namespace {

#if defined(_WIN32)
int FileDescriptor(FILE* stream) { return _fileno(stream); }
int Duplicate(int descriptor) { return _dup(descriptor); }
int Redirect(int source, int destination) { return _dup2(source, destination); }
int Close(int descriptor) { return _close(descriptor); }
#else
int FileDescriptor(FILE* stream) { return fileno(stream); }
int Duplicate(int descriptor) { return dup(descriptor); }
int Redirect(int source, int destination) { return dup2(source, destination); }
int Close(int descriptor) { return close(descriptor); }
#endif

class StreamCapture {
 public:
    explicit StreamCapture(FILE* stream) : stream_(stream), descriptor_(FileDescriptor(stream)) {
        file_ = std::tmpfile();
        if (file_ == nullptr) {
            throw std::runtime_error("failed to create CLI output capture file");
        }

        savedDescriptor_ = Duplicate(descriptor_);
        if (savedDescriptor_ < 0) {
            std::fclose(file_);
            file_ = nullptr;
            throw std::runtime_error("failed to duplicate CLI output descriptor");
        }

        std::fflush(stream_);
        if (Redirect(FileDescriptor(file_), descriptor_) < 0) {
            (void)Close(savedDescriptor_);
            std::fclose(file_);
            file_ = nullptr;
            savedDescriptor_ = -1;
            throw std::runtime_error("failed to redirect CLI output descriptor");
        }
    }

    ~StreamCapture() {
        if (!finished_) {
            std::fflush(stream_);
            if (savedDescriptor_ >= 0) {
                (void)Redirect(savedDescriptor_, descriptor_);
                (void)Close(savedDescriptor_);
            }
            if (file_ != nullptr) {
                std::fclose(file_);
            }
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

        const bool restored = Redirect(savedDescriptor_, descriptor_) >= 0;
        const bool descriptorClosed = Close(savedDescriptor_) == 0;
        const bool fileClosed = std::fclose(file_) == 0;
        finished_ = true;

        if (!restored || !descriptorClosed || !fileClosed) {
            throw std::runtime_error("failed to restore CLI output descriptor");
        }
        return output;
    }

 private:
    FILE* stream_;
    FILE* file_ = nullptr;
    int descriptor_;
    int savedDescriptor_ = -1;
    bool finished_ = false;
};

#if defined(_WIN32)
class EnvironmentGuard {
 public:
    explicit EnvironmentGuard(const wchar_t* name) : name_(name) {
        if (const wchar_t* value = _wgetenv(name)) {
            previous_ = value;
        }
    }

    ~EnvironmentGuard() {
        (void)_wputenv_s(name_.c_str(), previous_ ? previous_->c_str() : L"");
    }

    bool set(const std::wstring& value) const {
        return _wputenv_s(name_.c_str(), value.c_str()) == 0;
    }

 private:
    std::wstring name_;
    std::optional<std::wstring> previous_;
};

std::optional<std::wstring> EnvironmentValue(const wchar_t* name) {
    if (const wchar_t* value = _wgetenv(name)) {
        return value;
    }
    return std::nullopt;
}
#else
class EnvironmentGuard {
 public:
    explicit EnvironmentGuard(const char* name) : name_(name) {
        if (const char* value = std::getenv(name)) {
            previous_ = value;
        }
    }

    ~EnvironmentGuard() {
        if (previous_) {
            (void)setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            (void)unsetenv(name_.c_str());
        }
    }

    bool set(const std::string& value) const {
        return setenv(name_.c_str(), value.c_str(), 1) == 0;
    }

 private:
    std::string name_;
    std::optional<std::string> previous_;
};

std::optional<std::string> EnvironmentValue(const char* name) {
    if (const char* value = std::getenv(name)) {
        return value;
    }
    return std::nullopt;
}
#endif

class SettingsSandbox {
 public:
    SettingsSandbox()
#if defined(_WIN32)
        : appData_(L"APPDATA")
#elif defined(__APPLE__)
        : fixedHome_("CFFIXED_USER_HOME")
#else
        : home_("HOME"), xdgConfigHome_("XDG_CONFIG_HOME")
#endif
    {
        static std::atomic<unsigned long long> sequence = 0;
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ = std::filesystem::temp_directory_path() /
                ("cyberether-cli-" + std::to_string(nonce) + "-" +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));

        std::error_code ec;
        const bool created = std::filesystem::create_directory(root_, ec);
        if (ec || !created) {
            throw std::runtime_error("failed to create CLI test settings sandbox");
        }

        if (!ConfigureEnvironment(root_)) {
            std::filesystem::remove_all(root_, ec);
            throw std::runtime_error("failed to configure CLI test settings sandbox");
        }
    }

    ~SettingsSandbox() {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    const std::filesystem::path& root() const { return root_; }

 private:
    bool ConfigureEnvironment(const std::filesystem::path& root) {
#if defined(_WIN32)
        return appData_.set(root.wstring());
#elif defined(__APPLE__)
        return fixedHome_.set(root.string());
#else
        return home_.set(root.string()) && xdgConfigHome_.set(root.string());
#endif
    }

    std::filesystem::path root_;
#if defined(_WIN32)
    EnvironmentGuard appData_;
#elif defined(__APPLE__)
    EnvironmentGuard fixedHome_;
#else
    EnvironmentGuard home_;
    EnvironmentGuard xdgConfigHome_;
#endif
};

class LogLevelGuard {
 public:
    LogLevelGuard() : previous_(_JST_LOG_DEBUG_LEVEL()) {}
    ~LogLevelGuard() { JST_LOG_SET_DEBUG_LEVEL(previous_); }

 private:
    int previous_;
};

class SettingsGuard {
 public:
    SettingsGuard() {
        if (Jetstream::Settings::Get(previous_) != Jetstream::Result::SUCCESS) {
            throw std::runtime_error("failed to retain CLI test settings");
        }
    }

    ~SettingsGuard() { (void)Jetstream::Settings::Set(previous_, false); }

    const Jetstream::Settings& previous() const { return previous_; }

 private:
    Jetstream::Settings previous_;
};

struct InvocationResult {
    int code;
    std::string out;
    std::string err;
    int logLevelAfterRun;
    bool sandboxUntouched;
};

SettingsSandbox* settingsSandbox = nullptr;

InvocationResult Invoke(const std::vector<std::string>& arguments) {
    REQUIRE(settingsSandbox != nullptr);
    const auto& sandboxRoot = settingsSandbox->root();

    std::vector<std::string> values = {"cyberether"};
    values.insert(values.end(), arguments.begin(), arguments.end());

    std::vector<char*> argv;
    argv.reserve(values.size() + 1);
    for (auto& value : values) {
        argv.push_back(value.data());
    }
    argv.push_back(nullptr);

    LogLevelGuard logLevel;
    StreamCapture out(stdout);
    StreamCapture err(stderr);
    const int code = Jetstream::Run(static_cast<int>(values.size()), argv.data());
    const int logLevelAfterRun = _JST_LOG_DEBUG_LEVEL();
    const std::string capturedOut = out.finish();
    const std::string capturedErr = err.finish();

    std::error_code ec;
    const bool sandboxUntouched = std::filesystem::is_empty(sandboxRoot, ec) && !ec;
    return {code, capturedOut, capturedErr, logLevelAfterRun, sandboxUntouched};
}

InvocationResult Invoke(std::initializer_list<const char*> arguments) {
    std::vector<std::string> values;
    values.reserve(arguments.size());
    for (const char* argument : arguments) {
        values.emplace_back(argument);
    }
    return Invoke(values);
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
    CAPTURE(result.code, result.out, result.err);

    CHECK(result.code == code);
    CHECK(result.sandboxUntouched);
    if (code == 0) {
        CHECK_FALSE(result.out.empty());
        CHECK(result.err.empty());
    } else if (code == 2) {
        CHECK(result.out.empty());
        CHECK(result.err.starts_with("Error: "));
        CHECK(result.err.ends_with("Try 'cyberether --help' for more information.\n"));
    }

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

std::string UsageError(const std::string& message) {
    return "Error: " + message + "\nTry 'cyberether --help' for more information.\n";
}

void ExpectUsageError(const char* label,
                      std::initializer_list<const char*> arguments,
                      const std::string& message) {
    INFO("CLI case: " << label);
    const InvocationResult result = Invoke(arguments);
    CAPTURE(result.code, result.out, result.err);

    CHECK(result.code == 2);
    CHECK(result.out.empty());
    CHECK(result.err == UsageError(message));
    CHECK(result.sandboxUntouched);
}

void ExpectVersion(const char* label, std::initializer_list<const char*> arguments) {
    INFO("CLI case: " << label);
    const InvocationResult result = Invoke(arguments);
    CAPTURE(result.code, result.out, result.err);

    CHECK(result.code == 0);
    CHECK(result.err.empty());
    CHECK(result.sandboxUntouched);
    CHECK(result.out == std::string("CyberEther v") + JETSTREAM_VERSION_STR + "-" +
                            JETSTREAM_BUILD_TYPE + "\n");
}

}  // namespace

TEST_CASE("CLI displays contextual help and version", "[core][integration][cli]") {
    Expect("global help",
           {"--help"},
           0,
           {"Usage:\n",
            "cyberether [options] [flowgraph]",
            "Commands:\n",
            "Global Options:\n",
            "Graphics Options:\n",
            "CyberEther Remote Options:\n",
            "Benchmark Options:\n",
            "Examples:\n"},
           {},
           {},
           {"Error:"});
    Expect("short global help", {"-h"}, 0, {"cyberether [options] [flowgraph]"});
    Expect("run help",
           {"run", "--help"},
           0,
           {"run [options] [flowgraph]", "Global Options:\n", "Graphics Options:\n"},
           {},
           {"Commands:\n", "Benchmark Options:\n"});
    Expect("implicit run help",
           {"flowgraph.yaml", "--help"},
           0,
           {"run [options] [flowgraph]", "Graphics Options:\n"},
           {},
           {"Commands:\n", "Benchmark Options:\n"});
    Expect("benchmark help",
           {"benchmark", "--help"},
           0,
           {"benchmark [options] [block]", "Global Options:\n", "Benchmark Options:\n"},
           {},
           {"Commands:\n", "Graphics Options:\n", "CyberEther Remote Options:\n"});
    Expect("command ordering", {"-v", "run", "--help"}, 0, {"run [options] [flowgraph]"});

    ExpectVersion("version", {"--version"});
    ExpectVersion("short version", {"-V"});
}

TEST_CASE("CLI help and version obey left-to-right precedence", "[core][integration][cli]") {
    const InvocationResult globalHelp = Invoke({"--help"});
    const InvocationResult benchmarkHelp = Invoke({"benchmark", "--help"});
    const InvocationResult version = Invoke({"--version"});

    REQUIRE(globalHelp.code == 0);
    REQUIRE(benchmarkHelp.code == 0);
    REQUIRE(version.code == 0);

    const InvocationResult helpFirst = Invoke({"--help", "--unknown", "--version"});
    CHECK(helpFirst.code == 0);
    CHECK(helpFirst.out == globalHelp.out);
    CHECK(helpFirst.err.empty());
    CHECK(helpFirst.sandboxUntouched);

    const InvocationResult versionFirst = Invoke({"--version", "--unknown", "--help"});
    CHECK(versionFirst.code == 0);
    CHECK(versionFirst.out == version.out);
    CHECK(versionFirst.err.empty());
    CHECK(versionFirst.sandboxUntouched);

    const InvocationResult helpBeforePostParseConflict =
        Invoke({"--headless", "benchmark", "--help"});
    CHECK(helpBeforePostParseConflict.code == 0);
    CHECK(helpBeforePostParseConflict.out == benchmarkHelp.out);
    CHECK(helpBeforePostParseConflict.err.empty());
    CHECK(helpBeforePostParseConflict.sandboxUntouched);

    ExpectUsageError("unknown before help",
                     {"--unknown", "--help"},
                     "Unknown option: '--unknown'.");
    ExpectUsageError("missing value before help",
                     {"--scale", "--help"},
                     "Missing value for --scale. Expected a positive number.");
}

TEST_CASE("CLI accepts every documented enum value", "[core][integration][cli]") {
    struct EnumCase {
        const char* label;
        std::vector<std::string> arguments;
        const char* currentValue;
    };

    const std::vector<EnumCase> cases = {
        {"renderer metal", {"--renderer=METAL", "--help"},
         "Preferred graphics backend (current: metal)"},
        {"renderer vulkan", {"--renderer=VuLkAn", "--help"},
         "Preferred graphics backend (current: vulkan)"},
        {"codec h264", {"--remote", "--codec=H264", "--help"},
         "Streaming codec (current: h264)"},
        {"codec av1", {"--remote", "--codec=AV1", "--help"},
         "Streaming codec (current: av1)"},
        {"codec vp8", {"--remote", "--codec=VP8", "--help"},
         "Streaming codec (current: vp8)"},
        {"codec vp9", {"--remote", "--codec=VP9", "--help"},
         "Streaming codec (current: vp9)"},
        {"encoder auto", {"--remote", "--encoder=AUTO", "--help"},
         "Streaming encoder (current: auto)"},
        {"encoder software", {"--remote", "--encoder=SOFTWARE", "--help"},
         "Streaming encoder (current: software)"},
        {"encoder nvenc", {"--remote", "--encoder=NVENC", "--help"},
         "Streaming encoder (current: nvenc)"},
        {"encoder v4l2", {"--remote", "--encoder=V4L2", "--help"},
         "Streaming encoder (current: v4l2)"},
        {"encoder videotoolbox", {"--remote", "--encoder=VIDEOTOOLBOX", "--help"},
         "Streaming encoder (current: videotoolbox)"},
        {"encoder mediafoundation", {"--remote", "--encoder=MEDIAFOUNDATION", "--help"},
         "Streaming encoder (current: mediafoundation)"},
        {"format markdown", {"benchmark", "--format=MARKDOWN", "--help"},
         "Output format (current: markdown)"},
        {"format json", {"benchmark", "--format=JSON", "--help"},
         "Output format (current: json)"},
        {"format csv", {"benchmark", "--format=CSV", "--help"},
         "Output format (current: csv)"},
    };

    for (const auto& entry : cases) {
        INFO("CLI enum case: " << entry.label);
        const InvocationResult result = Invoke(entry.arguments);
        CAPTURE(result.code, result.out, result.err);
        CHECK(result.code == 0);
        CHECK(result.out.find(entry.currentValue) != std::string::npos);
        CHECK(result.err.empty());
        CHECK(result.sandboxUntouched);
    }
}

TEST_CASE("CLI repeated options use last scalar and idempotent flag values",
          "[core][integration][cli]") {
    Expect("repeated run options",
           {"--renderer=metal",
            "--renderer",
            "vulkan",
            "--device-index=1",
            "--device-index",
            "2",
            "--size=320x240",
            "--size",
            "800X600",
            "--scale=1",
            "--scale",
            "2.5",
            "--framerate=30",
            "--framerate",
            "75",
            "--remote",
            "--remote",
            "--broker=https://first.example",
            "--broker",
            "https://second.example/path?key=value",
            "--codec=h264",
            "--codec",
            "vp9",
            "--encoder=auto",
            "--encoder",
            "software",
            "--headless",
            "--headless",
            "--auto-join-sessions",
            "--auto-join-sessions",
            "--plugin=first.cep",
            "--plugin",
            "second.CEP",
            "--help"},
           0,
           {"Preferred graphics backend (current: vulkan)",
            "Vulkan and CUDA device index (current: 2)",
            "Viewport size (current: 800x600)",
            "Interface scale factor (current: 2.5)",
            "Target frame rate (current: 75)",
            "Broker URL (current: https://second.example/path?key=value)",
            "Streaming codec (current: vp9)",
            "Streaming encoder (current: software)"});
    Expect("repeated benchmark format",
           {"benchmark", "--format=markdown", "--format", "csv", "--help"},
           0,
           {"Output format (current: csv)"});
}

TEST_CASE("CLI accepts normalized, inline, and boundary values", "[core][integration][cli]") {
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
    Expect("numeric boundaries",
           {"--device-index=18446744073709551615",
            "--size=2147483647X1",
            "--scale=0.125",
            "--framerate=18446744073709551615",
            "--help"},
           0,
            {"Vulkan and CUDA device index (current: 18446744073709551615)",
             "Viewport size (current: 2147483647x1)",
             "Interface scale factor (current: 0.125)",
             "Target frame rate (current: 18446744073709551615)"});
    Expect("minimum numeric boundaries",
           {"--device-index=0", "--size=1X1", "--scale=0.125", "--framerate=1", "--help"},
           0,
           {"Vulkan and CUDA device index (current: 0)",
            "Viewport size (current: 1x1)",
            "Interface scale factor (current: 0.125)",
            "Target frame rate (current: 1)"});
    Expect("fresh settings after overrides",
           {"--help"},
           0,
           {"Preferred graphics backend (current: automatic)",
            "Vulkan and CUDA device index (current: 0)",
            "Viewport size (current: 1920x1080)",
            "Interface scale factor (current: 1.0)",
            "Target frame rate (current: 60)",
            "Broker URL (current: https://cyberether.org)"});
}

TEST_CASE("CLI rejects invalid syntax and command conflicts", "[core][integration][cli]") {
    Expect("malformed delimiter value", {"--=value", "--help"}, 2, {}, {"Unknown option: '--=value'."});
    Expect("empty malformed delimiter", {"--=", "--help"}, 2, {}, {"Unknown option: '--='."});
    Expect("unknown long option", {"--unknown"}, 2, {}, {"Unknown option: '--unknown'."});
    Expect("unknown short option", {"-x"}, 2, {}, {"Unknown option: '-x'."});
    Expect("dash-prefixed flowgraph", {"--", "--flowgraph.yml", "second.yml"}, 2, {}, {"Only one flowgraph"});
    Expect("benchmark delimiter", {"benchmark", "--", "--block", "second"}, 2, {}, {"Only one benchmark block"});
    Expect("multiple flowgraphs", {"one.yml", "two.yml"}, 2, {}, {"Only one flowgraph"});
    Expect("multiple benchmark blocks", {"benchmark", "fft", "am"}, 2, {}, {"Only one benchmark block"});
    Expect("run option with benchmark", {"benchmark", "--headless"}, 2, {}, {"not available for the benchmark command"});
    Expect("run option before benchmark", {"--renderer", "metal", "benchmark"}, 2, {}, {"not available for the benchmark command"});
    Expect("device index with benchmark",
           {"benchmark", "--device-index", "7", "--help"},
           0,
           {"Vulkan and CUDA device index (current: 7)"});
    Expect("benchmark option with run", {"run", "--format", "json"}, 2, {}, {"only available for the benchmark command"});
    Expect("benchmark option before run", {"--format=csv", "run"}, 2, {}, {"only available for the benchmark command"});
}

TEST_CASE("CLI keeps command and delimiter parser boundaries deterministic",
          "[core][integration][cli]") {
    ExpectUsageError("command after flowgraph",
                     {"graph.yaml", "benchmark", "--help"},
                     "Only one flowgraph may be provided; received 'benchmark'.");
    ExpectUsageError("second explicit run command",
                     {"run", "run", "second.yaml"},
                     "Only one flowgraph may be provided; received 'second.yaml'.");
    ExpectUsageError("run token after benchmark block",
                     {"benchmark", "fft", "run"},
                     "Only one benchmark block may be provided; received 'run'.");
    ExpectUsageError("command token after delimiter",
                     {"--", "benchmark", "second.yaml"},
                     "Only one flowgraph may be provided; received 'second.yaml'.");
    ExpectUsageError("benchmark command token after delimiter",
                     {"benchmark", "--", "run", "second"},
                     "Only one benchmark block may be provided; received 'second'.");
    ExpectUsageError("bare dash remains option syntax",
                     {"-", "second.yaml"},
                     "Unknown option: '-'.");
    ExpectUsageError("short options are not grouped",
                     {"-vh", "--help"},
                     "Unknown option: '-vh'.");
    ExpectUsageError("inline value retains additional separators",
                     {"--renderer=metal=vulkan", "--help"},
                     "Invalid value for --renderer: 'metal=vulkan'. Expected: metal or vulkan.");
}

TEST_CASE("CLI enforces option values and dependencies", "[core][integration][cli]") {
    const std::vector<std::pair<const char*, const char*>> errors = {
        {"--plugin", "Missing value for --plugin"},
        {"--plugin=", "Missing value for --plugin"},
        {"--renderer", "Missing value for --renderer"},
        {"--renderer=software", "Invalid value for --renderer"},
        {"--broker", "Missing value for --broker"},
        {"--broker=", "Missing value for --broker"},
        {"--codec", "Missing value for --codec"},
        {"--codec=invalid", "Invalid value for --codec"},
        {"--encoder", "Missing value for --encoder"},
        {"--encoder=invalid", "Invalid value for --encoder"},
        {"--format", "Missing value for --format"},
        {"--format=xml", "Invalid value for --format"},
        {"--help=true", "does not accept a value"},
        {"--version=true", "does not accept a value"},
        {"--headless=true", "does not accept a value"},
        {"--remote=true", "does not accept a value"},
        {"--auto-join-sessions=true", "does not accept a value"},
    };
    for (const auto& [argument, message] : errors) {
        Expect(argument, {argument}, 2, {}, {message});
    }

    Expect("broker dependency", {"--broker", "https://example.com"}, 2, {}, {"requires --remote"});
    Expect("codec dependency", {"--codec", "h264"}, 2, {}, {"requires --remote"});
    Expect("encoder dependency", {"--encoder", "auto"}, 2, {}, {"requires --remote"});
    Expect("auto join dependency", {"--auto-join-sessions"}, 2, {}, {"requires --remote"});
}

TEST_CASE("CLI validates numeric values", "[core][integration][cli]") {
    const std::vector<std::pair<const char*, const char*>> errors = {
        {"--size", "Missing value for --size"},
        {"--size=", "Missing value for --size"},
        {"--size=0x480", "Invalid value for --size"},
        {"--size=640x0", "Invalid value for --size"},
        {"--size=640x480x2", "Invalid value for --size"},
        {"--size=+640x480", "Invalid value for --size"},
        {"--size=2147483648x480", "Invalid value for --size"},
        {"--size=640x2147483648", "Invalid value for --size"},
        {"--size=18446744073709551616x480", "Invalid value for --size"},
        {"--scale", "Missing value for --scale"},
        {"--scale=", "Missing value for --scale"},
        {"--scale=0", "Invalid value for --scale"},
        {"--scale=-1", "Invalid value for --scale"},
        {"--scale=1.0x", "Invalid value for --scale"},
        {"--scale=nan", "Invalid value for --scale"},
        {"--scale=inf", "Invalid value for --scale"},
        {"--scale=1e1000", "Invalid value for --scale"},
        {"--framerate", "Missing value for --framerate"},
        {"--framerate=", "Missing value for --framerate"},
        {"--framerate=0", "Invalid value for --framerate"},
        {"--framerate=-1", "Invalid value for --framerate"},
        {"--framerate=1.5", "Invalid value for --framerate"},
        {"--framerate=18446744073709551616", "Invalid value for --framerate"},
        {"--device-index", "Missing value for --device-index"},
        {"--device-index=", "Missing value for --device-index"},
        {"--device-index=-1", "Invalid value for --device-index"},
        {"--device-index=+1", "Invalid value for --device-index"},
        {"--device-index=invalid", "Invalid value for --device-index"},
        {"--device-index=18446744073709551616", "Invalid value for --device-index"},
    };
    for (const auto& [argument, message] : errors) {
        Expect(argument, {argument}, 2, {}, {message});
    }
}

TEST_CASE("CLI diagnoses separate negative numeric values as invalid",
          "[core][integration][cli]") {
    struct NegativeCase {
        const char* option;
        const char* value;
        const char* message;
    };

    const std::vector<NegativeCase> cases = {
        {"--device-index", "-1",
         "Invalid value for --device-index: '-1'. Expected a non-negative integer."},
        {"--size", "-1x480",
         "Invalid value for --size: '-1x480'. Expected dimensions from 1 to 2147483647."},
        {"--scale", "-1", "Invalid value for --scale: '-1'. Expected a positive number."},
        {"--framerate", "-1",
         "Invalid value for --framerate: '-1'. Expected a positive integer."},
    };

    for (const auto& entry : cases) {
        INFO("CLI negative value case: " << entry.option);
        const InvocationResult result = Invoke(
            std::vector<std::string>{entry.option, entry.value});
        CAPTURE(result.code, result.out, result.err);
        CHECK(result.code == 2);
        CHECK(result.out.empty());
        CHECK(result.sandboxUntouched);

        // Expected failure: dash-prefixed values are mistaken for missing values.
        CHECK(result.err == UsageError(entry.message));
    }
}

TEST_CASE("CLI rejects removed commands and options", "[core][integration][cli]") {
    Expect("remote command", {"remote"}, 2, {}, {"Unknown command: 'remote'."});
    Expect("device option", {"--device"}, 2, {}, {"Unknown option: '--device'."});
    Expect("device ID option", {"--device-id"}, 2, {}, {"Unknown option: '--device-id'."});
    Expect("endpoint option", {"--endpoint"}, 2, {}, {"Unknown option: '--endpoint'."});
    Expect("auto join option", {"--auto-join"}, 2, {}, {"Unknown option: '--auto-join'."});
}

TEST_CASE("CLI validates plugin paths before startup", "[core][integration][cli]") {
    const InvocationResult result = Invoke({"--plugin=archive.tar.gz", "--help"});
    CAPTURE(result.code, result.out, result.err);
    CHECK(result.sandboxUntouched);

    // Expected failure: .cep extension validation is deferred until plugin startup.
    CHECK(result.code == 2);
    if (result.code == 2) {
        CHECK(result.out.empty());
        CHECK(result.err == UsageError(
            "Invalid value for --plugin: 'archive.tar.gz'. Expected a .cep path."));
    }
}

TEST_CASE("CLI restores the process log level after parser exits",
          "[core][integration][cli]") {
    LogLevelGuard restoreLogLevel;
    constexpr int sentinel = -17;
    const std::vector<std::vector<std::string>> cases = {
        {"--help"},
        {"-v", "--help"},
        {"-vv", "--version"},
    };

    for (const auto& arguments : cases) {
        JST_LOG_SET_DEBUG_LEVEL(sentinel);
        const InvocationResult result = Invoke(arguments);
        CAPTURE(arguments, result.code, result.out, result.err, result.logLevelAfterRun);
        REQUIRE(result.code == 0);

        REQUIRE(result.logLevelAfterRun == sentinel);
        REQUIRE(_JST_LOG_DEBUG_LEVEL() == sentinel);
    }
}

TEST_CASE("CLI reports invalid retained remote settings before backend startup",
          "[core][integration][cli]") {
    SettingsGuard settings;
    Jetstream::Settings invalid = settings.previous();
    invalid.remote.codec = "not-a-codec";
    REQUIRE(Jetstream::Settings::Set(invalid, false) == Jetstream::Result::SUCCESS);

    const InvocationResult result = Invoke({"--remote"});
    CAPTURE(result.code, result.out, result.err);
    CHECK(result.code == 2);
    CHECK(result.err == UsageError(
        "The configured CyberEther Remote codec or encoder is invalid."));
    CHECK(result.sandboxUntouched);

    // Expected failure: remote enum conversion logs before returning the usage error.
    CHECK(result.out.empty());
}

TEST_CASE("CLI settings sandbox restores environment variables",
          "[core][integration][cli]") {
#if defined(_WIN32)
    const auto appData = EnvironmentValue(L"APPDATA");
    {
        SettingsSandbox nested;
        CHECK(EnvironmentValue(L"APPDATA") ==
              std::optional<std::wstring>(nested.root().wstring()));
    }
    CHECK(EnvironmentValue(L"APPDATA") == appData);
#elif defined(__APPLE__)
    const auto fixedHome = EnvironmentValue("CFFIXED_USER_HOME");
    {
        SettingsSandbox nested;
        CHECK(EnvironmentValue("CFFIXED_USER_HOME") ==
              std::optional<std::string>(nested.root().string()));
    }
    CHECK(EnvironmentValue("CFFIXED_USER_HOME") == fixedHome);
#else
    const auto home = EnvironmentValue("HOME");
    const auto xdgConfigHome = EnvironmentValue("XDG_CONFIG_HOME");
    {
        SettingsSandbox nested;
        const auto nestedRoot = std::optional<std::string>(nested.root().string());
        CHECK(EnvironmentValue("HOME") == nestedRoot);
        CHECK(EnvironmentValue("XDG_CONFIG_HOME") == nestedRoot);
    }
    CHECK(EnvironmentValue("HOME") == home);
    CHECK(EnvironmentValue("XDG_CONFIG_HOME") == xdgConfigHome);
#endif
}

// TODO: Inject Run dependencies for Backend::Configure<CUDA>,
// Backend::Initialize<CPU>, Backend::DestroyAll, and Plugin::Load, plus fakeable
// Instance, Flowgraph, and Remote factories. Controllable create,
// flowgraphCreate/importFromFile, start, Remote::create, poll, stop, and destroy
// results are required to assert pluginDestroy, Instance::stop/destroy, and
// Backend::DestroyAll ordering after every acquired resource without touching
// drivers, windows, hardware, or the network.

TEST_CASE("CLI rejects broker URLs unsupported by remote transport", "[core][integration][cli]") {
    const InvocationResult result = Invoke({"--remote", "--broker=ftp://example.com", "--help"});
    CAPTURE(result.code, result.out, result.err);
    CHECK(result.sandboxUntouched);

    // Expected failure: broker scheme validation is deferred until network startup.
    CHECK(result.code == 2);
    CHECK(result.out.empty());
    CHECK(result.err.find("Invalid value for --broker") != std::string::npos);
}

int main(int argc, char* argv[]) {
    SettingsSandbox sandbox;
    settingsSandbox = &sandbox;
    const int code = Catch::Session().run(argc, argv);
    settingsSandbox = nullptr;
    return code;
}
