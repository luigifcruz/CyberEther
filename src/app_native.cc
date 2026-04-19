#include <algorithm>
#include <cstdio>
#include <thread>

#include "jetstream/app.hh"
#include "jetstream/config.hh"
#include "jetstream/detail/instance_remote_supervisor.hh"
#include "jetstream/instance.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/benchmark.hh"

namespace Jetstream {

namespace {

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

}  // namespace

void printUsage(const char* program) {
    const std::string codecOptions = RemoteCodecOptionsString();
    const std::string encoderOptions = RemoteEncoderOptionsString();

    jst::fmt::print("Usage: {} [command] [options] [flowgraph]\n\n", program);
    jst::fmt::print("Commands:\n");
    jst::fmt::print("  run                          Run normally (default)\n");
    jst::fmt::print("  remote                       Run with remote streaming enabled\n");
    jst::fmt::print("  benchmark                    Run benchmarks\n\n");
    jst::fmt::print("Global Options:\n");
    jst::fmt::print("  -h, --help                   Show this help\n");
    jst::fmt::print("  -v, -vv                      Increase log verbosity (debug, trace)\n");
    jst::fmt::print("  -V, --version                Show version\n\n");
    jst::fmt::print("Graphics Options:\n");
    jst::fmt::print("  --device <type>              Device type (metal, vulkan)\n");
    jst::fmt::print("  --device-id <id>             Device ID (default: 0)\n");
    jst::fmt::print("  --headless                   Run in headless mode (no window)\n");
    jst::fmt::print("  --size <WxH>                 Window size (default: 1920x1080)\n");
    jst::fmt::print("  --scale <scale>              Interface scale factor (default: 1.0)\n");
    jst::fmt::print("  --framerate <fps>            Target framerate (default: 60)\n\n");
    jst::fmt::print("Benchmark Options:\n");
    jst::fmt::print("  --format <type>             Output format: markdown, json, csv (default: markdown)\n\n");
    jst::fmt::print("Remote Options:\n");
    jst::fmt::print("  --endpoint <url>             Broker URL (default: https://cyberether.org)\n");
    jst::fmt::print("  --auto-join                  Auto-join sessions\n");
    jst::fmt::print("  --codec <codec>              Codec: {} (default: {})\n",
                    codecOptions,
                    GetRemoteCodecName(Instance::Remote::CodecType::H264));
    jst::fmt::print("  --encoder <type>             Encoder: {} (default: {})\n",
                    encoderOptions,
                    GetRemoteEncoderName(Instance::Remote::EncoderType::Auto));
}

enum class CommandType {
    Run,
    Remote,
    Benchmark,
};

int RunAppNative(int argc, char* argv[], PluginCreateFn pluginCreate, PluginDestroyFn pluginDestroy) {
    CommandType command = CommandType::Run;

    Instance::Config config = {
        .compositor = CompositorType::DEFAULT,
    };
    Instance::Remote::Config remoteConfig;

    std::string flowgraphPath;
    std::string benchmarkFormat = "markdown";

    std::shared_ptr<Instance> instance;
    std::shared_ptr<Flowgraph> flowgraph;
    std::unique_ptr<Instance::Remote::Supervisor> supervisor;

    std::atomic<int> code{0};

    //
    // Parse Arguments
    //

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];

        // Handle Commands

        if (i == 1) {
            if (arg == "run") {
                command = CommandType::Run;
                continue;
            }

            if (arg == "remote") {
                command = CommandType::Remote;
                continue;
            }

            if (arg == "benchmark") {
                command = CommandType::Benchmark;
                continue;
            }
        }

        // Handle Global Options

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }

        if (arg == "-V" || arg == "--version") {
            jst::fmt::print("CyberEther v{}-{}\n", JETSTREAM_VERSION_STR, JETSTREAM_BUILD_TYPE);
            return 0;
        }

        if (arg == "-v") {
            JST_LOG_SET_DEBUG_LEVEL(std::max(_JST_LOG_DEBUG_LEVEL(), 3));
            continue;
        }

        if (arg == "-vv") {
            JST_LOG_SET_DEBUG_LEVEL(std::max(_JST_LOG_DEBUG_LEVEL(), 4));
            continue;
        }

        // Handle Graphics Options

        if (arg == "--device") {
            if (i + 1 < argc) {
                config.device = StringToDevice(argv[++i]);
            }
            continue;
        }

        if (arg == "--device-id") {
            if (i + 1 < argc) {
                ++i; // TODO: Implement device selection.
            }
            continue;
        }

        if (arg == "--headless") {
            config.headless = true;
            continue;
        }

        if (arg == "--size") {
            if (i + 1 < argc) {
                std::string sizeStr = argv[++i];
                size_t xPos = sizeStr.find('x');
                if (xPos != std::string::npos) {
                    config.size.x = std::stoull(sizeStr.substr(0, xPos));
                    config.size.y = std::stoull(sizeStr.substr(xPos + 1));
                }
            }
            continue;
        }

        if (arg == "--scale") {
            if (i + 1 < argc) {
                config.scale = std::stof(argv[++i]);
            }
            continue;
        }

        if (arg == "--framerate") {
            if (i + 1 < argc) {
                config.framerate = std::stoull(argv[++i]);
            }
            continue;
        }

        // Handle Command Options

        switch (command) {
            case CommandType::Benchmark:
                if (arg == "--format") {
                    if (i + 1 >= argc) {
                        jst::fmt::print(stderr,
                                        "Missing value for --format. Expected one of: markdown, json, csv.\n\n");
                        printUsage(argv[0]);
                        return -1;
                    }

                    benchmarkFormat = argv[++i];
                    if (benchmarkFormat != "markdown" && benchmarkFormat != "json" && benchmarkFormat != "csv") {
                        jst::fmt::print(stderr,
                                        "Invalid value for --format: '{}'. Expected one of: markdown, json, csv.\n\n",
                                        benchmarkFormat);
                        printUsage(argv[0]);
                        return -1;
                    }
                    continue;
                }
                break;

            case CommandType::Remote:
                if (arg == "--endpoint") {
                    if (i + 1 < argc) {
                        remoteConfig.broker = argv[++i];
                    }
                    continue;
                }

                if (arg == "--codec") {
                    if (i + 1 < argc) {
                        const std::string codec = argv[++i];
                        try {
                            remoteConfig.codec = StringToRemoteCodec(codec);
                        } catch (const Result&) {
                            jst::fmt::print(stderr,
                                            "Invalid value for --codec: '{}'. Expected one of: {}.\n\n",
                                            codec,
                                            RemoteCodecOptionsString());
                            printUsage(argv[0]);
                            return -1;
                        }
                    } else {
                        jst::fmt::print(stderr,
                                        "Missing value for --codec. Expected one of: {}.\n\n",
                                        RemoteCodecOptionsString());
                        printUsage(argv[0]);
                        return -1;
                    }
                    continue;
                }

                if (arg == "--encoder") {
                    if (i + 1 < argc) {
                        const std::string enc = argv[++i];
                        try {
                            remoteConfig.encoder = StringToRemoteEncoder(enc);
                        } catch (const Result&) {
                            jst::fmt::print(stderr,
                                            "Invalid value for --encoder: '{}'. Expected one of: {}.\n\n",
                                            enc,
                                            RemoteEncoderOptionsString());
                            printUsage(argv[0]);
                            return -1;
                        }
                    } else {
                        jst::fmt::print(stderr,
                                        "Missing value for --encoder. Expected one of: {}.\n\n",
                                        RemoteEncoderOptionsString());
                        printUsage(argv[0]);
                        return -1;
                    }
                    continue;
                }

                if (arg == "--auto-join") {
                    remoteConfig.autoJoinSessions = true;
                    continue;
                }
                break;

            case CommandType::Run:
                break;
        }

        if (arg[0] == '-') {
            jst::fmt::print(stderr, "Unknown option: '{}'.\n\n", arg);
            printUsage(argv[0]);
            return -1;
        }

        if (command == CommandType::Benchmark) {
            jst::fmt::print(stderr, "The benchmark command does not accept a flowgraph path: '{}'.\n\n", arg);
            printUsage(argv[0]);
            return -1;
        }

        if (arg[0] != '-') {
            flowgraphPath = arg;
        }
    }

    JST_INFO("[CYBERETHER] Running native app.");

    //
    // Benchmark Logic
    //

    if (command == CommandType::Benchmark) {
        Benchmark::Run(benchmarkFormat);
        return 0;
    }

    //
    // Interface Logic
    //

    if (command == CommandType::Run || command == CommandType::Remote) {
        instance = std::make_shared<Instance>();

        if (instance->create(config) != Result::SUCCESS) {
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

        if (command == CommandType::Remote) {
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

        if (command == CommandType::Remote && instance->remote()) {
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

        if (command == CommandType::Remote && instance->remote()->started()) {
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
