#include <atomic>
#include <cstdio>
#include <thread>

#include "jetstream/app.hh"
#include "jetstream/config.hh"
#include "jetstream/instance.hh"
#include "jetstream/backend/base.hh"

#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/wasmfs.h>

namespace Jetstream {

static std::atomic<int> code{0};
static std::shared_ptr<Instance> instance;
static std::thread computeThread;

int RunAppBrowser() {
    JST_INFO("[CYBERETHER] Running browser app.");

    std::thread([] {
        backend_t opfs = wasmfs_create_opfs_backend();
        int ret = wasmfs_create_directory("/storage", 0777, opfs);
        JST_DEBUG("OPFS mount on /storage: {}", ret == 0 ? "OK" : "FAILED");
    }).detach();

    instance = std::make_shared<Instance>();
    code.store(0);

    Instance::Config config = {
        .compositor = CompositorType::DEFAULT,
    };

    if (instance->create(config) != Result::SUCCESS) {
        instance.reset();
        return -1;
    }

    if (instance->start() != Result::SUCCESS) {
        (void)instance->destroy();
        instance.reset();
        return -1;
    }

    computeThread = std::thread([&]{
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

    auto graphicalThreadLoop = [](void* arg) {
        Instance* instance = reinterpret_cast<Instance*>(arg);
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
            StopAppBrowser();
        }
    };

    emscripten_set_main_loop_arg(graphicalThreadLoop, instance.get(), 0, 1);

    return 0;
}

int StopAppBrowser() {
    if (!instance) {
        return 0;
    }

    JST_INFO("[CYBERETHER] Stopping app browser.");

    emscripten_cancel_main_loop();

    if (instance->computing() || instance->presenting()) {
        (void)instance->stop();
    }

    if (computeThread.joinable()) {
        computeThread.join();
    }

    (void)instance->destroy();
    instance.reset();

    Backend::DestroyAll();

    return code.load();
}

}  // namespace Jetstream
