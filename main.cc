#include <thread>

#include "jetstream/config.hh"
#include "jetstream/instance.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/benchmark.hh"
#include "jetstream/instance_remote_ui.hh"

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/wasmfs.h>
#endif

using namespace Jetstream;

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
    jst::fmt::print("Usage: {} [command] [options] [flowgraph]\n\n", program);
    jst::fmt::print("Commands:\n");
    jst::fmt::print("  remote                       Run with remote streaming enabled\n");
    jst::fmt::print("  benchmark [format]           Run benchmarks (markdown, json, csv)\n\n");
    jst::fmt::print("Global Options:\n");
    jst::fmt::print("  -d, --device <type>          Device type (metal, vulkan)\n");
    jst::fmt::print("  --device-id <id>             Device ID (default: 0)\n");
    jst::fmt::print("  -h, --help                   Show this help\n");
    jst::fmt::print("  -v, --version                Show version\n\n");
    jst::fmt::print("Remote Options:\n");
    jst::fmt::print("  --headless                   Run in headless mode (no window)\n");
    jst::fmt::print("  --size <WxH>                 Window size (default: 1920x1080)\n");
    jst::fmt::print("  --framerate <fps>            Target framerate (default: 60)\n");
    jst::fmt::print("  --endpoint <url>             Broker URL (default: https://cyberether.org)\n");
    jst::fmt::print("  --codec <codec>              Video codec: h264, vp8, vp9, av1 (default: h264)\n");
    jst::fmt::print("  --acceleration               Enable hardware acceleration\n");
    jst::fmt::print("  --auto-join                  Auto-join sessions\n");
}

int main(int argc, char* argv[]) {
    Instance::Config config = {
        .compositor = CompositorType::DEFAULT,
    };
    std::string flowgraphPath;
    bool enableRemote = false;
    Instance::Remote::Config remoteConfig;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }

        if (arg == "-v" || arg == "--version") {
            jst::fmt::print("CyberEther v{}-{}\n", JETSTREAM_VERSION_STR, JETSTREAM_BUILD_TYPE);
            return 0;
        }

        if (arg == "-d" || arg == "--device") {
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

        if (arg == "benchmark") {
            std::string format = "markdown";
            if (i + 1 < argc) {
                const std::string next = argv[i + 1];
                if (next == "markdown" || next == "json" || next == "csv") {
                    format = next;
                    ++i;
                }
            }
            Benchmark::Run(format);
            return 0;
        }

        if (arg == "remote") {
            enableRemote = true;
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

        if (arg == "--framerate") {
            if (i + 1 < argc) {
                config.framerate = std::stoull(argv[++i]);
            }
            continue;
        }

        if (arg == "--endpoint") {
            if (i + 1 < argc) {
                remoteConfig.broker = argv[++i];
            }
            continue;
        }

        if (arg == "--codec") {
            if (i + 1 < argc) {
                const std::string codec = argv[++i];
                if (codec == "h264") {
                    remoteConfig.codec = Viewport::VideoCodec::H264;
                } else if (codec == "vp8") {
                    remoteConfig.codec = Viewport::VideoCodec::VP8;
                } else if (codec == "vp9") {
                    remoteConfig.codec = Viewport::VideoCodec::VP9;
                } else if (codec == "av1") {
                    remoteConfig.codec = Viewport::VideoCodec::AV1;
                }
            }
            continue;
        }

        if (arg == "--acceleration") {
            remoteConfig.hardwareAcceleration = true;
            continue;
        }

        if (arg == "--auto-join") {
            remoteConfig.autoJoinSessions = true;
            continue;
        }

        if (arg[0] != '-') {
            flowgraphPath = arg;
        }
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

    if (enableRemote) {
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
    std::unique_ptr<RemoteSessionMonitor> sessionMonitor;

    if (enableRemote && instance->remote()) {
        PrintRemoteInfo(instance->remote().get());

        sessionMonitor = std::make_unique<RemoteSessionMonitor>(
            instance->remote().get(),
            remoteConfig.autoJoinSessions,
            PromptRemoteClientApproval
        );
        sessionMonitor->start();
    }

    while (instance->polling()) {
        instance->poll();
    }

    if (sessionMonitor) {
        sessionMonitor->stop();
    }
#endif

    if (enableRemote && instance->remote()->started()) {
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
