#include <thread>
#include <iostream>
#include <atomic>
#include <set>

#include "jetstream/config.hh"
#include "jetstream/instance.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/benchmark.hh"

#include <qrencode.h>

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

void PrintRemoteInfo(Instance::Remote* remote) {
    QRcode* qr = QRcode_encodeString8bit(remote->inviteUrl().c_str(), 0, QR_ECLEVEL_L);
    if (!qr) {
        jst::fmt::print(stderr, "[QR encode error]\n");
        return;
    }

    const int qrWidth = qr->width;
    const int border = 2;
    const int totalWidth = qrWidth + border * 2;
    const int boxInner = totalWidth + 4;

    auto isBlack = [&](int x, int y) -> bool {
        if (x < 0 || y < 0 || x >= qrWidth || y >= qrWidth) return false;
        return (qr->data[y * qrWidth + x] & 1) != 0;
    };

    std::string hLine;
    for (int i = 0; i < boxInner; ++i) hLine += "═";

    auto printCentered = [&](const std::string& text) {
        int totalPad = boxInner - static_cast<int>(text.length());
        int left = totalPad / 2;
        int right = totalPad - left;
        jst::fmt::print("║{:>{}}{}{:>{}}\n", "", left, text, "║", right + 1);
    };

    jst::fmt::print("\n╔{}╗\n", hLine);
    printCentered("CyberEther Remote");
    printCentered("Scan QR code or open link to connect");
    jst::fmt::print("╠{}╣\n", hLine);

    for (int y = -border; y < qrWidth + border; y += 2) {
        std::string row;
        for (int x = -border; x < qrWidth + border; ++x) {
            const bool upper = isBlack(x, y);
            const bool lower = isBlack(x, y + 1);
            if (upper && lower)      row += "█";
            else if (upper)          row += "▀";
            else if (lower)          row += "▄";
            else                     row += " ";
        }
        jst::fmt::print("║  {}  ║\n", row);
    }

    jst::fmt::print("╚{}╝\n\n", hLine);

    QRcode_free(qr);

    jst::fmt::print("Room ID:      {}\n", remote->roomId());
    jst::fmt::print("Join URL:     {}\n", remote->inviteUrl());
    jst::fmt::print("Access Token: {}\n\n", remote->accessToken());
}

void PrintClientApproval(const std::string& code) {
    const int boxInner = 38;

    std::string hLine;
    for (int i = 0; i < boxInner; ++i) hLine += "═";

    auto printCentered = [&](const std::string& text) {
        int totalPad = boxInner - static_cast<int>(text.length());
        int left = totalPad / 2;
        int right = totalPad - left;
        jst::fmt::print("║{:>{}}{}{:>{}}\n", "", left, text, "║", right + 1);
    };

    jst::fmt::print("\n╔{}╗\n", hLine);
    printCentered("New Connection Request");
    printCentered("Verify client code before approving");
    jst::fmt::print("╠{}╣\n", hLine);
    printCentered(code);
    jst::fmt::print("╚{}╝\n\n", hLine);

    jst::fmt::print("Approve? [Y/n]: ");
    std::fflush(stdout);
}

void RunSessionMonitor(Instance::Remote* remote, std::atomic<bool>& running, bool autoJoin) {
    std::set<std::string> seenSessions;

    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        if (!running) break;

        const auto& waitlist = remote->waitlist();
        if (waitlist.empty()) {
            continue;
        }

        for (const auto& sessionId : waitlist) {
            if (seenSessions.count(sessionId)) continue;
            seenSessions.insert(sessionId);

            std::string code = sessionId.substr(sessionId.length() - 6);
            std::transform(code.begin(), code.end(), code.begin(), ::toupper);

            if (autoJoin) {
                remote->approveClient(code);
            } else {
                PrintClientApproval(code);

                std::string input;
                if (std::getline(std::cin, input)) {
                    if (input.empty() || input == "y" || input == "Y") {
                        remote->approveClient(code);
                    }
                }
            }
        }
    }
}

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
    jst::fmt::print("  --endpoint <url>             Broker URL (default: https://api.cyberether.org)\n");
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
    std::atomic<bool> sessionMonitorRunning{false};
    std::thread sessionMonitorThread;

    if (enableRemote && instance->remote()) {
        PrintRemoteInfo(instance->remote().get());

        sessionMonitorRunning = true;
        sessionMonitorThread = std::thread(RunSessionMonitor,
            instance->remote().get(),
            std::ref(sessionMonitorRunning),
            remoteConfig.autoJoinSessions);
    }

    while (instance->polling()) {
        instance->poll();
    }

    sessionMonitorRunning = false;
    if (sessionMonitorThread.joinable()) {
        sessionMonitorThread.join();
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
