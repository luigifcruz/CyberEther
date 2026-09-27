#include <atomic>
#include <cstdio>
#include <string>
#include <thread>

#include "jetstream/run.hh"
#include "jetstream/config.hh"
#include "jetstream/instance.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/platform.hh"
#include "jetstream/plugin.hh"
#include "jetstream/settings.hh"

#include <emscripten.h>
#include <emscripten/html5.h>

namespace Jetstream {

static std::atomic<bool> shutdownRequested{false};
static std::shared_ptr<Instance> instance;
static std::thread computeThread;
static std::string startupTheme;
static std::string startupFlowgraph;

static void OnWebGPUInitialized(const Result webgpuResult) {
    if (webgpuResult != Result::SUCCESS ||
        shutdownRequested.load(std::memory_order_acquire)) {
        Backend::WebGPU::CancelInitialization();
        return;
    }

    Settings settings;
    if (Settings::Get(settings) != Result::SUCCESS) {
        JST_WARN("[CYBERETHER] Failed to load settings. Using defaults.");
        settings = {};
        (void)Settings::Set(settings, false);
    }
    const Settings retainedSettings = settings;

    if (!startupTheme.empty()) {
        settings.interface.themeKey = startupTheme;
        if (Settings::Set(settings, false) != Result::SUCCESS) {
            JST_WARN("[CYBERETHER] Failed to apply startup theme.");
        }
    }

    for (const auto& path : settings.registry.plugins) {
        if (Plugin::Load(path) != Result::SUCCESS) {
            JST_WARN("[CYBERETHER] Failed to load plugin '{}'. Continuing startup.", path);
        }
    }

    instance = std::make_shared<Instance>();
    Instance::Config config = {
        .compositor = CompositorType::DEFAULT,
        .pythonRuntimePath = settings.runtime.python.path,
        .dependencyPolicy = settings.runtime.dependencyPolicy,
    };

    const Result createResult = instance->create(config);

    const Result restoreResult = Settings::Set(retainedSettings, false);
    if (createResult != Result::SUCCESS || restoreResult != Result::SUCCESS) {
        if (createResult == Result::SUCCESS) {
            (void)instance->destroy();
        }
        instance.reset();
        Backend::DestroyAll();
        return;
    }

    if (instance->start() != Result::SUCCESS) {
        (void)instance->destroy();
        instance.reset();
        Backend::DestroyAll();
        return;
    }

    if (!startupFlowgraph.empty()) {
        std::shared_ptr<Viewport::Generic> viewport;
        if (instance->viewportGet(viewport) != Result::SUCCESS || !viewport ||
            viewport->addFileDropEvent({startupFlowgraph}) != Result::SUCCESS) {
            JST_WARN("[CYBERETHER] Failed to queue startup flowgraph '{}'.", startupFlowgraph);
        }
    }

    computeThread = std::thread([&] {
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
                RequestShutdown();
                break;
            }
        }
    });

    auto graphicalThreadLoop = [](void* arg) {
        Instance* currentInstance = reinterpret_cast<Instance*>(arg);
        Result res = Result::SUCCESS;

        if (!shutdownRequested.load(std::memory_order_acquire)) {
            try {
                res = currentInstance->present();
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
        }

        if (res != Result::SUCCESS && res != Result::RELOAD) {
            RequestShutdown();
        } else if (!currentInstance->presenting()) {
            RequestShutdown();
        }

        if (!shutdownRequested.load(std::memory_order_acquire)) {
            return;
        }

        JST_INFO("[CYBERETHER] Stopping browser app.");

        emscripten_cancel_main_loop();

        if (currentInstance->computing() || currentInstance->presenting()) {
            (void)currentInstance->stop();
        }

        if (computeThread.joinable()) {
            computeThread.join();
        }

        (void)currentInstance->destroy();
        instance.reset();

        Backend::DestroyAll();
    };

    emscripten_set_main_loop_arg(graphicalThreadLoop, instance.get(), 0, 0);
}

int Run(int argc, char* argv[]) {
    JST_INFO("[CYBERETHER] Running browser app.");

    startupTheme.clear();
    startupFlowgraph.clear();
    bool positionalOnly = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (!positionalOnly && arg == "--") {
            positionalOnly = true;
            continue;
        }

        if (!positionalOnly && (arg == "--theme" || arg.starts_with("--theme="))) {
            if (arg == "--theme") {
                if (++i >= argc || !argv[i][0] || std::string(argv[i]).starts_with("-")) {
                    JST_ERROR("[CYBERETHER] Missing value for --theme.");
                    return -1;
                }
                startupTheme = argv[i];
            } else {
                startupTheme = arg.substr(8);
                if (startupTheme.empty()) {
                    JST_ERROR("[CYBERETHER] Missing value for --theme.");
                    return -1;
                }
            }
            continue;
        }

        if (!positionalOnly && arg.starts_with("-")) {
            JST_ERROR("[CYBERETHER] Unknown browser option '{}'.", arg);
            return -1;
        }
        if (!startupFlowgraph.empty()) {
            JST_ERROR("[CYBERETHER] Only one flowgraph may be provided; received '{}'.", arg);
            return -1;
        }
        startupFlowgraph = arg;
    }

    shutdownRequested.store(false);

    if (Platform::InitializePersistentStorage() != Result::SUCCESS) {
        return -1;
    }

    if (Backend::WebGPU::InitializeAsync(OnWebGPUInitialized) != Result::SUCCESS) {
        return -1;
    }

    return 0;
}

void RequestShutdown() {
    shutdownRequested.store(true, std::memory_order_release);
}

}  // namespace Jetstream
