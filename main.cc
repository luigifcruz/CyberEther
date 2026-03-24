#include <cstdio>
#include <thread>

#include "jetstream/config.hh"
#include "jetstream/detail/instance_remote_supervisor.hh"
#include "jetstream/instance.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/benchmark.hh"

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/wasmfs.h>
#endif

using namespace Jetstream;

namespace {

std::string RemoteCodecOptionsString() {
    std::string options;

    for (const auto codec : Jetstream::RemoteCodecTypes) {
        if (!options.empty()) {
            options += ", ";
        }

        options += Jetstream::GetRemoteCodecName(codec);
    }

    return options;
}

std::string RemoteEncoderOptionsString() {
    std::string options;

    for (const auto encoder : Jetstream::RemoteEncoderTypes) {
        if (!options.empty()) {
            options += ", ";
        }

        options += Jetstream::GetRemoteEncoderName(encoder);
    }

    return options;
}

}  // namespace

#ifdef JST_OS_BROWSER
extern "C" {
EMSCRIPTEN_KEEPALIVE
void cyberether_shutdown() {
    JST_INFO("Shutting down...");
    emscripten_cancel_main_loop();
    emscripten_runtime_keepalive_pop();
    emscripten_force_exit(0);
}
}
#endif

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
    jst::fmt::print("  -v, --version                Show version\n\n");
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
                    Jetstream::GetRemoteCodecName(Instance::Remote::CodecType::H264));
    jst::fmt::print("  --encoder <type>             Encoder: {} (default: {})\n",
                    encoderOptions,
                    Jetstream::GetRemoteEncoderName(Instance::Remote::EncoderType::Auto));
}

enum class CommandType {
    Run,
    Remote,
    Benchmark,
};

int main(int argc, char* argv[]) {
    CommandType command = CommandType::Run;

    Instance::Config config = {
        .compositor = CompositorType::DEFAULT,
    };
    Instance::Remote::Config remoteConfig;

    std::string flowgraphPath;
    std::string benchmarkFormat = "markdown";

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

        if (arg == "-v" || arg == "--version") {
            jst::fmt::print("CyberEther v{}-{}\n", JETSTREAM_VERSION_STR, JETSTREAM_BUILD_TYPE);
            return 0;
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
                        return 1;
                    }

                    benchmarkFormat = argv[++i];
                    if (benchmarkFormat != "markdown" && benchmarkFormat != "json" && benchmarkFormat != "csv") {
                        jst::fmt::print(stderr,
                                        "Invalid value for --format: '{}'. Expected one of: markdown, json, csv.\n\n",
                                        benchmarkFormat);
                        printUsage(argv[0]);
                        return 1;
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
                            remoteConfig.codec = Jetstream::StringToRemoteCodec(codec);
                        } catch (const Result&) {
                            jst::fmt::print(stderr,
                                            "Invalid value for --codec: '{}'. Expected one of: {}.\n\n",
                                            codec,
                                            RemoteCodecOptionsString());
                            printUsage(argv[0]);
                            return 1;
                        }
                    } else {
                        jst::fmt::print(stderr,
                                        "Missing value for --codec. Expected one of: {}.\n\n",
                                        RemoteCodecOptionsString());
                        printUsage(argv[0]);
                        return 1;
                    }
                    continue;
                }

                if (arg == "--encoder") {
                    if (i + 1 < argc) {
                        const std::string enc = argv[++i];
                        try {
                            remoteConfig.encoder = Jetstream::StringToRemoteEncoder(enc);
                        } catch (const Result&) {
                            jst::fmt::print(stderr,
                                            "Invalid value for --encoder: '{}'. Expected one of: {}.\n\n",
                                            enc,
                                            RemoteEncoderOptionsString());
                            printUsage(argv[0]);
                            return 1;
                        }
                    } else {
                        jst::fmt::print(stderr,
                                        "Missing value for --encoder. Expected one of: {}.\n\n",
                                        RemoteEncoderOptionsString());
                        printUsage(argv[0]);
                        return 1;
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
            return 1;
        }

        if (command == CommandType::Benchmark) {
            jst::fmt::print(stderr, "The benchmark command does not accept a flowgraph path: '{}'.\n\n", arg);
            printUsage(argv[0]);
            return 1;
        }

        if (arg[0] != '-') {
            flowgraphPath = arg;
        }
    }

    if (command == CommandType::Benchmark) {
        Benchmark::Run(benchmarkFormat);
        return 0;
    }

#ifdef JST_OS_BROWSER
    std::thread([] {
        backend_t opfs = wasmfs_create_opfs_backend();
        int ret = wasmfs_create_directory("/storage", 0777, opfs);
        JST_DEBUG("OPFS mount on /storage: {}", ret == 0 ? "OK" : "FAILED");
    }).detach();
#endif

    auto instance = std::make_shared<Instance>();

    JST_CHECK_THROW(instance->create(config));

    if (!flowgraphPath.empty()) {
        std::shared_ptr<Flowgraph> flowgraph;
        JST_CHECK_THROW(instance->flowgraphCreate("main", {}, flowgraph));
        JST_CHECK_THROW(flowgraph->importFromFile(flowgraphPath));
    }

    JST_CHECK_THROW(instance->start());

    if (command == CommandType::Remote) {
        JST_CHECK_THROW(instance->remote()->create(remoteConfig));
    }

    auto computeThread = std::thread([&]{
        while (instance->computing()) {
            JST_CHECK_THROW(instance->compute());
        }
    });

    auto graphicalThreadLoop = [](void* arg) {
        Instance* instance = reinterpret_cast<Instance*>(arg);
        JST_CHECK_THROW(instance->present());
    };

#ifdef JST_OS_BROWSER
    emscripten_set_main_loop_arg(graphicalThreadLoop, instance.get(), 0, 1);
#else
    auto graphicalThread = std::thread([&]{
        while (instance->presenting()) {
            graphicalThreadLoop(instance.get());
        }
    });
#endif

#ifdef JST_OS_BROWSER
    emscripten_runtime_keepalive_push();
#else
    std::unique_ptr<Instance::Remote::Supervisor> supervisor;

    if (command == CommandType::Remote && instance->remote()) {
        supervisor = std::make_unique<Instance::Remote::Supervisor>(
            instance->remote().get(),
            remoteConfig.autoJoinSessions
        );
        supervisor->print();
        supervisor->start();
    }

    while (instance->polling()) {
        instance->poll();
    }

    if (supervisor) {
        supervisor->stop();
    }
#endif

    if (command == CommandType::Remote && instance->remote()->started()) {
        JST_CHECK_THROW(instance->remote()->destroy());
    }

    JST_CHECK_THROW(instance->stop());

    if (computeThread.joinable()) {
        computeThread.join();
    }

#ifdef JST_OS_BROWSER
    emscripten_cancel_main_loop();
#else
    if (graphicalThread.joinable()) {
        graphicalThread.join();
    }
#endif

    JST_CHECK_THROW(instance->destroy());

    Jetstream::Backend::DestroyAll();

    return 0;
}
