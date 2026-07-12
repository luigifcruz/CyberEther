#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <string_view>
#include <thread>

#include "jetstream/run.hh"
#include "jetstream/config.hh"
#include "jetstream/detail/instance_remote_supervisor.hh"
#include "jetstream/instance.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/plugin.hh"
#include "jetstream/registry.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/benchmark.hh"

namespace Jetstream {

namespace {

Instance::Config BuildInstanceConfig(const Settings& settings) {
    Instance::Config config = {
        .device = settings.graphics.device,
        .compositor = CompositorType::DEFAULT,
        .headless = settings.graphics.headless,
        .size = {settings.graphics.size.width, settings.graphics.size.height},
        .scale = settings.graphics.scale,
        .framerate = settings.graphics.framerate,
        .pythonRuntimePath = settings.runtime.python.path,
    };

    return config;
}

Instance::Remote::Config BuildRemoteConfig(const Settings& settings) {
    Instance::Remote::Config config = {
        .broker = settings.remote.brokerUrl,
        .autoJoinSessions = settings.remote.autoJoinSessions,
        .framerate = static_cast<U32>(settings.remote.framerate),
        .encoder = StringToRemoteEncoder(settings.remote.encoder),
        .codec = StringToRemoteCodec(settings.remote.codec),
    };

    return config;
}

void LoadRegistryPlugins(const Settings& settings) {
    for (const auto& path : settings.registry.plugins) {
        if (Plugin::Load(path) != Result::SUCCESS) {
            JST_WARN("[CYBERETHER] Failed to load plugin '{}'. Continuing startup.", path);
        }
    }
}

std::string RemoteCodecOptionsString() {
    std::string options;

    for (const auto codec : RemoteCodecTypes) {
        if (!options.empty()) {
            options += ", ";
        }

        options += GetRemoteCodecName(codec);
    }

    return options;
}

std::string RemoteEncoderOptionsString() {
    std::string options;

    for (const auto encoder : RemoteEncoderTypes) {
        if (!options.empty()) {
            options += ", ";
        }

        options += GetRemoteEncoderName(encoder);
    }

    return options;
}

enum class CommandType {
    Run,
    Benchmark,
};

constexpr int CLI_USAGE_ERROR = 2;
constexpr U64 MAX_VIEWPORT_DIMENSION = static_cast<U64>(std::numeric_limits<I32>::max());

std::string Lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool IsRemoteCodec(const std::string& value) {
    return std::any_of(RemoteCodecTypes.begin(), RemoteCodecTypes.end(), [&](const auto codec) {
        return value == GetRemoteCodecName(codec);
    });
}

bool IsRemoteEncoder(const std::string& value) {
    return std::any_of(RemoteEncoderTypes.begin(), RemoteEncoderTypes.end(), [&](const auto encoder) {
        return value == GetRemoteEncoderName(encoder);
    });
}

bool ParsePositiveInteger(const std::string& value, U64& result) {
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isdigit(c);
    })) {
        return false;
    }

    try {
        size_t parsed = 0;
        const U64 number = std::stoull(value, &parsed);
        if (parsed != value.size() || number == 0) {
            return false;
        }
        result = number;
    } catch (const std::exception&) {
        return false;
    }

    return true;
}

bool ParsePositiveFloat(const std::string& value, F32& result) {
    try {
        size_t parsed = 0;
        const F32 number = std::stof(value, &parsed);
        if (parsed != value.size() || !std::isfinite(number) || number <= 0.0f) {
            return false;
        }
        result = number;
    } catch (const std::exception&) {
        return false;
    }

    return true;
}

int PrintUsageError(const char* program, const std::string& message) {
    jst::fmt::print(stderr, "Error: {}\nTry '{} --help' for more information.\n", message, program);
    return CLI_USAGE_ERROR;
}

}  // namespace

static void printUsage(const char* program,
                       const Settings& settings,
                       const std::optional<CommandType> command = std::nullopt) {
    const std::string codecOptions = RemoteCodecOptionsString();
    const std::string encoderOptions = RemoteEncoderOptionsString();
    const std::string renderer = settings.graphics.device.has_value()
        ? GetDeviceName(*settings.graphics.device)
        : "automatic";
    std::string interfaceScale = jst::fmt::format("{}", settings.graphics.scale);
    if (std::isfinite(settings.graphics.scale) && interfaceScale.find_first_of(".eE") == std::string::npos) {
        interfaceScale += ".0";
    }

    std::string encoderChoicesFirst = encoderOptions;
    std::string encoderChoicesSecond;
    const size_t encoderWrap = encoderOptions.find(", videotoolbox");
    if (encoderWrap != std::string::npos) {
        encoderChoicesFirst = encoderOptions.substr(0, encoderWrap + 1);
        encoderChoicesSecond = encoderOptions.substr(encoderWrap + 2);
    }

    jst::fmt::print("Usage:\n");
    if (!command.has_value()) {
        jst::fmt::print("  {} [options] [flowgraph]\n", program);
        jst::fmt::print("  {} run [options] [flowgraph]\n", program);
        jst::fmt::print("  {} benchmark [options] [block]\n\n", program);
        jst::fmt::print("Commands:\n");
        jst::fmt::print("  run                          Launch CyberEther (default)\n");
        jst::fmt::print("  benchmark                    Run performance benchmarks\n\n");
    } else if (*command == CommandType::Benchmark) {
        jst::fmt::print("  {} benchmark [options] [block]\n\n", program);
    } else {
        jst::fmt::print("  {} run [options] [flowgraph]\n\n", program);
    }

    jst::fmt::print("Global Options:\n");
    jst::fmt::print("  -h, --help                   Show this help\n");
    jst::fmt::print("  -v, -vv                      Set debug or trace log level\n");
    jst::fmt::print("  -V, --version                Show version\n\n");

    if (!command.has_value() || *command == CommandType::Run) {
        jst::fmt::print("Graphics Options:\n");
        jst::fmt::print("  --renderer <renderer>        Preferred graphics backend (current: {})\n", renderer);
        jst::fmt::print("  {:<29}  Choices: metal, vulkan\n", "");
        jst::fmt::print("  --headless                   Run without a window\n");
        jst::fmt::print("  --size <WxH>                 Viewport size (current: {}x{})\n",
                        settings.graphics.size.width,
                        settings.graphics.size.height);
        jst::fmt::print("  --scale <factor>             Interface scale factor (current: {})\n",
                        interfaceScale);
        jst::fmt::print("  --framerate <fps>            Target frame rate (current: {})\n\n",
                        settings.graphics.framerate);
        jst::fmt::print("CyberEther Remote Options:\n");
        jst::fmt::print("  --remote                     Enable CyberEther Remote\n");
        jst::fmt::print("  --broker <url>               Broker URL (current: {})\n",
                        settings.remote.brokerUrl);
        jst::fmt::print("  --auto-join-sessions         Automatically join remote sessions\n");
        jst::fmt::print("  --codec <codec>              Streaming codec (current: {})\n",
                        settings.remote.codec);
        jst::fmt::print("  {:<29}  Choices: {}\n", "", codecOptions);
        jst::fmt::print("  --encoder <encoder>          Streaming encoder (current: {})\n",
                        settings.remote.encoder);
        jst::fmt::print("  {:<29}  Choices: {}\n", "", encoderChoicesFirst);
        if (!encoderChoicesSecond.empty()) {
            jst::fmt::print("  {:<29}           {}\n", "", encoderChoicesSecond);
        }
    }

    if (!command.has_value() || *command == CommandType::Benchmark) {
        if (!command.has_value()) {
            jst::fmt::print("\n");
        }
        jst::fmt::print("Benchmark Options:\n");
        jst::fmt::print("  --format <format>            Output format (current: {})\n",
                        settings.benchmark.format);
        jst::fmt::print("  {:<29}  Choices: markdown, json, csv\n", "");
    }

    jst::fmt::print("\nExamples:\n");
    if (!command.has_value() || *command == CommandType::Run) {
        jst::fmt::print("  {} flowgraph.yaml\n", program);
        jst::fmt::print("  {} --remote flowgraph.yaml\n", program);
    }
    if (!command.has_value() || *command == CommandType::Benchmark) {
        jst::fmt::print("  {} benchmark fft --format json\n", program);
    }
}

int Run(int argc, char* argv[], PluginCreateFn pluginCreate, PluginDestroyFn pluginDestroy) {
    CommandType command = CommandType::Run;
    bool commandSelected = false;
    bool remoteEnabled = false;

    Settings settings;
    if (Settings::Get(settings) != Result::SUCCESS) {
        JST_WARN("[CYBERETHER] Failed to load settings. Using defaults.");
        settings = {};
        (void)Settings::Set(settings, false);
    }
    const Settings retainedSettings = settings;

    JST_LOG_SET_DEBUG_LEVEL(settings.developer.logLevel);

    std::string flowgraphPath;

    std::shared_ptr<Instance> instance;
    std::shared_ptr<Flowgraph> flowgraph;
    std::unique_ptr<Instance::Remote::Supervisor> supervisor;

    std::atomic<int> code{0};

    //
    // Parse Arguments
    //

    std::string runOption;
    std::string remoteSettingOption;
    std::string benchmarkOption;
    std::string benchmarkFilter;
    bool positionalOnly = false;

    for (int i = 1; i < argc; i++) {
        const std::string originalArg = argv[i];
        std::string arg = originalArg;
        std::optional<std::string> inlineValue;

        if (!positionalOnly && originalArg != "--" && arg.starts_with("--")) {
            const size_t separator = arg.find('=');
            if (separator != std::string::npos) {
                inlineValue = arg.substr(separator + 1);
                arg = arg.substr(0, separator);
            }
        }

        auto takeValue = [&](std::string& value) {
            if (inlineValue.has_value()) {
                if (inlineValue->empty()) {
                    return false;
                }
                value = *inlineValue;
                return true;
            }

            if (i + 1 >= argc || std::string_view(argv[i + 1]).starts_with("-")) {
                return false;
            }

            value = argv[++i];
            return !value.empty();
        };

        auto rejectInlineValue = [&]() {
            return inlineValue.has_value()
                ? PrintUsageError(argv[0], jst::fmt::format("Option '{}' does not accept a value.", arg))
                : 0;
        };

        if (!positionalOnly && originalArg == "--") {
            positionalOnly = true;
            continue;
        }

        // Handle Commands

        if (!positionalOnly && flowgraphPath.empty() && !commandSelected) {
            if (arg == "run") {
                command = CommandType::Run;
                commandSelected = true;
                continue;
            }

            if (arg == "benchmark") {
                command = CommandType::Benchmark;
                commandSelected = true;
                continue;
            }

            if (arg == "remote") {
                return PrintUsageError(argv[0], "Unknown command: 'remote'.");
            }

        }

        // Handle Global Options

        if (!positionalOnly && (arg == "-h" || arg == "--help")) {
            if (const int error = rejectInlineValue()) {
                return error;
            }
            std::optional<CommandType> helpCommand;
            if (commandSelected || !flowgraphPath.empty()) {
                helpCommand = command;
            } else if (!runOption.empty()) {
                helpCommand = CommandType::Run;
            } else if (!benchmarkOption.empty()) {
                helpCommand = CommandType::Benchmark;
            }
            printUsage(argv[0], settings, helpCommand);
            return 0;
        }

        if (!positionalOnly && (arg == "-V" || arg == "--version")) {
            if (const int error = rejectInlineValue()) {
                return error;
            }
            jst::fmt::print("CyberEther v{}-{}\n", JETSTREAM_VERSION_STR, JETSTREAM_BUILD_TYPE);
            return 0;
        }

        if (!positionalOnly && arg == "-v") {
            settings.developer.logLevel = 3;
            JST_LOG_SET_DEBUG_LEVEL(settings.developer.logLevel);
            continue;
        }

        if (!positionalOnly && arg == "-vv") {
            settings.developer.logLevel = 4;
            JST_LOG_SET_DEBUG_LEVEL(settings.developer.logLevel);
            continue;
        }

        // Handle Graphics Options

        if (!positionalOnly && arg == "--renderer") {
            std::string value;
            if (!takeValue(value)) {
                return PrintUsageError(argv[0], "Missing value for --renderer. Expected: metal or vulkan.");
            }
            value = Lowercase(value);
            if (value != "metal" && value != "vulkan") {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Invalid value for --renderer: '{}'. Expected: metal or vulkan.", value));
            }
            settings.graphics.device = StringToDevice(value);
            runOption = arg;
            continue;
        }

        if (!positionalOnly && arg == "--headless") {
            if (const int error = rejectInlineValue()) {
                return error;
            }
            settings.graphics.headless = true;
            runOption = arg;
            continue;
        }

        if (!positionalOnly && arg == "--size") {
            std::string value;
            if (!takeValue(value)) {
                return PrintUsageError(argv[0], "Missing value for --size. Expected: WIDTHxHEIGHT.");
            }
            const size_t separator = value.find_first_of("xX");
            U64 width = 0;
            U64 height = 0;
            if (separator == std::string::npos ||
                value.find_first_of("xX", separator + 1) != std::string::npos ||
                !ParsePositiveInteger(value.substr(0, separator), width) ||
                !ParsePositiveInteger(value.substr(separator + 1), height) ||
                width > MAX_VIEWPORT_DIMENSION ||
                height > MAX_VIEWPORT_DIMENSION) {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Invalid value for --size: '{}'. Expected dimensions from 1 to {}.",
                    value,
                    MAX_VIEWPORT_DIMENSION));
            }
            settings.graphics.size.width = width;
            settings.graphics.size.height = height;
            runOption = arg;
            continue;
        }

        if (!positionalOnly && arg == "--scale") {
            std::string value;
            if (!takeValue(value)) {
                return PrintUsageError(argv[0], "Missing value for --scale. Expected a positive number.");
            }
            F32 scale = 0.0f;
            if (!ParsePositiveFloat(value, scale)) {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Invalid value for --scale: '{}'. Expected a positive number.", value));
            }
            settings.graphics.scale = scale;
            runOption = arg;
            continue;
        }

        if (!positionalOnly && arg == "--framerate") {
            std::string value;
            if (!takeValue(value)) {
                return PrintUsageError(argv[0], "Missing value for --framerate. Expected a positive integer.");
            }
            U64 framerate = 0;
            if (!ParsePositiveInteger(value, framerate)) {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Invalid value for --framerate: '{}'. Expected a positive integer.", value));
            }
            settings.graphics.framerate = framerate;
            runOption = arg;
            continue;
        }

        // Handle CyberEther Remote Options

        if (!positionalOnly && arg == "--remote") {
            if (const int error = rejectInlineValue()) {
                return error;
            }
            remoteEnabled = true;
            runOption = arg;
            continue;
        }

        if (!positionalOnly && arg == "--broker") {
            std::string value;
            if (!takeValue(value)) {
                return PrintUsageError(argv[0], "Missing value for --broker. Expected a URL.");
            }
            settings.remote.brokerUrl = value;
            runOption = arg;
            remoteSettingOption = arg;
            continue;
        }

        if (!positionalOnly && arg == "--codec") {
            std::string value;
            if (!takeValue(value)) {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Missing value for --codec. Expected one of: {}.", RemoteCodecOptionsString()));
            }
            value = Lowercase(value);
            if (!IsRemoteCodec(value)) {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Invalid value for --codec: '{}'. Expected one of: {}.",
                    value,
                    RemoteCodecOptionsString()));
            }
            settings.remote.codec = value;
            runOption = arg;
            remoteSettingOption = arg;
            continue;
        }

        if (!positionalOnly && arg == "--encoder") {
            std::string value;
            if (!takeValue(value)) {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Missing value for --encoder. Expected one of: {}.", RemoteEncoderOptionsString()));
            }
            value = Lowercase(value);
            if (!IsRemoteEncoder(value)) {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Invalid value for --encoder: '{}'. Expected one of: {}.",
                    value,
                    RemoteEncoderOptionsString()));
            }
            settings.remote.encoder = value;
            runOption = arg;
            remoteSettingOption = arg;
            continue;
        }

        if (!positionalOnly && arg == "--auto-join-sessions") {
            if (const int error = rejectInlineValue()) {
                return error;
            }
            settings.remote.autoJoinSessions = true;
            runOption = arg;
            remoteSettingOption = arg;
            continue;
        }

        // Handle Benchmark Options

        if (!positionalOnly && arg == "--format") {
            std::string value;
            if (!takeValue(value)) {
                return PrintUsageError(argv[0], "Missing value for --format. Expected one of: markdown, json, csv.");
            }
            value = Lowercase(value);
            if (value != "markdown" && value != "json" && value != "csv") {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Invalid value for --format: '{}'. Expected one of: markdown, json, csv.", value));
            }
            settings.benchmark.format = value;
            benchmarkOption = arg;
            continue;
        }

        if (!positionalOnly && arg.starts_with("-")) {
            return PrintUsageError(argv[0], jst::fmt::format("Unknown option: '{}'.", originalArg));
        }

        if (command == CommandType::Benchmark) {
            if (!benchmarkFilter.empty()) {
                return PrintUsageError(argv[0], jst::fmt::format(
                    "Only one benchmark block may be provided; received '{}'.", originalArg));
            }
            benchmarkFilter = Lowercase(originalArg);
            continue;
        }

        if (!flowgraphPath.empty()) {
            return PrintUsageError(argv[0], jst::fmt::format(
                "Only one flowgraph may be provided; received '{}'.", originalArg));
        }

        flowgraphPath = originalArg;
    }

    if (command == CommandType::Benchmark && !runOption.empty()) {
        return PrintUsageError(argv[0], jst::fmt::format(
            "Option '{}' is not available for the benchmark command.", runOption));
    }

    if (command == CommandType::Run && !benchmarkOption.empty()) {
        return PrintUsageError(argv[0], jst::fmt::format(
            "Option '{}' is only available for the benchmark command.", benchmarkOption));
    }

    if (!remoteEnabled && !remoteSettingOption.empty()) {
        return PrintUsageError(argv[0], jst::fmt::format(
            "Option '{}' requires --remote.", remoteSettingOption));
    }

    Instance::Remote::Config remoteConfig;
    if (remoteEnabled) {
        try {
            remoteConfig = BuildRemoteConfig(settings);
        } catch (const Result&) {
            return PrintUsageError(argv[0], "The configured CyberEther Remote codec or encoder is invalid.");
        }
    }

    JST_INFO("[CYBERETHER] Running native app.");

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    const auto backendConfig = Backend::Config {
        .headless = settings.graphics.headless,
        .pythonRuntimePath = settings.runtime.python.path,
    };
    if (Backend::Initialize<DeviceType::CPU>(backendConfig) != Result::SUCCESS) {
        return -1;
    }
#endif

    LoadRegistryPlugins(settings);

    //
    // Benchmark Logic
    //

    if (command == CommandType::Benchmark) {
        if (!benchmarkFilter.empty() &&
            Registry::ListAvailableBenchmarks(benchmarkFilter).empty()) {
            return PrintUsageError(argv[0], jst::fmt::format(
                "No benchmarks found for block '{}'.", benchmarkFilter));
        }
        Benchmark::Run(settings.benchmark.format, benchmarkFilter);
        return 0;
    }

    //
    // Interface Logic
    //

    if (command == CommandType::Run) {
        const Instance::Config config = BuildInstanceConfig(settings);

        if (Settings::Set(settings, false) != Result::SUCCESS) {
            return -1;
        }

        instance = std::make_shared<Instance>();

        const Result createResult = instance->create(config);

        // The compositor initializes from the effective CLI settings. Restore
        // retained settings before later UI changes can persist CLI overrides.
        if (Settings::Set(retainedSettings, false) != Result::SUCCESS) {
            return -1;
        }

        if (createResult != Result::SUCCESS) {
            return -1;
        }

        if (pluginCreate) {
            pluginCreate(instance.get());
        }

        if (!flowgraphPath.empty()) {
            if (instance->flowgraphCreate("main", {}, flowgraph) != Result::SUCCESS) {
                if (pluginDestroy) {
                    pluginDestroy(instance.get());
                }
                (void)instance->destroy();
                return -1;
            }
            if (flowgraph->importFromFile(flowgraphPath) != Result::SUCCESS) {
                if (pluginDestroy) {
                    pluginDestroy(instance.get());
                }
                (void)instance->destroy();
                return -1;
            }
        }

        if (instance->start() != Result::SUCCESS) {
            if (pluginDestroy) {
                pluginDestroy(instance.get());
            }
            (void)instance->destroy();
            return -1;
        }

        if (remoteEnabled) {
            if (instance->remote()->create(remoteConfig) != Result::SUCCESS) {
                (void)instance->stop();
                if (pluginDestroy) {
                    pluginDestroy(instance.get());
                }
                (void)instance->destroy();
                return -1;
            }
        }

        auto computeThread = std::thread([&]{
            while (instance->computing()) {
                Result res = Result::SUCCESS;

                try {
                    res = instance->compute();
                } catch (const Result& status) {
                    res = status;
                    JST_ERROR("[CYBERETHER] Compute loop exception: {}", status);
                } catch (const std::exception& e) {
                    res = Result::ERROR;
                    JST_ERROR("[CYBERETHER] Compute loop exception: {}", e.what());
                } catch (...) {
                    res = Result::ERROR;
                    JST_ERROR("[CYBERETHER] Unknown compute loop exception.");
                }

                if (res != Result::SUCCESS && res != Result::RELOAD) {
                    code.store(-1);
                    (void)instance->stop();
                    break;
                }
            }
        });

        auto graphicalThread = std::thread([&]{
            while (instance->presenting()) {
                Result res = Result::SUCCESS;

                try {
                    res = instance->present();
                } catch (const Result& status) {
                    res = status;
                    JST_ERROR("[CYBERETHER] Present loop exception: {}", status);
                } catch (const std::exception& e) {
                    res = Result::ERROR;
                    JST_ERROR("[CYBERETHER] Present loop exception: {}", e.what());
                } catch (...) {
                    res = Result::ERROR;
                    JST_ERROR("[CYBERETHER] Unknown present loop exception.");
                }

                if (res != Result::SUCCESS && res != Result::RELOAD) {
                    code.store(-1);
                    (void)instance->stop();
                    break;
                }
            }
        });

        if (remoteEnabled && instance->remote()) {
            supervisor = std::make_unique<Instance::Remote::Supervisor>(
                instance->remote().get(),
                remoteConfig.autoJoinSessions
            );
            supervisor->print();
            supervisor->start();
        }

        while (instance->polling()) {
            if (instance->poll() != Result::SUCCESS) {
                code.store(-1);
                break;
            }
        }

        if (supervisor) {
            supervisor->stop();
        }

        if (remoteEnabled && instance->remote()->started()) {
            (void)instance->remote()->destroy();
        }

        if (instance->computing() || instance->presenting()) {
            (void)instance->stop();
        }

        if (computeThread.joinable()) {
            computeThread.join();
        }

        if (graphicalThread.joinable()) {
            graphicalThread.join();
        }

        if (pluginDestroy) {
            pluginDestroy(instance.get());
        }

        (void)instance->destroy();

        Backend::DestroyAll();

        return code.load();
    }

    JST_ERROR("[CYBERETHER] Internal error occurred.");
    return -1;
}

}  // namespace Jetstream
