#include <atomic>
#include <cstdio>
#include <thread>

#include "jetstream/run.hh"
#include "jetstream/config.hh"
#include "jetstream/instance.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/platform.hh"
#include "jetstream/settings.hh"

#include <emscripten.h>
#include <emscripten/html5.h>

namespace Jetstream {

static std::atomic<bool> shutdownRequested{false};
static std::shared_ptr<Instance> instance;
static std::thread computeThread;

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

    instance = std::make_shared<Instance>();
    Instance::Config config = {
        .compositor = CompositorType::DEFAULT,
        .pythonRuntimePath = settings.runtime.python.path,
    };

    if (instance->create(config) != Result::SUCCESS) {
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

int Run() {
    JST_INFO("[CYBERETHER] Running browser app.");

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
