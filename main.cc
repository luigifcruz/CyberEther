#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

int main(int argc, char* argv[]) {
    std::cout << "Welcome to CyberEther!" << std::endl;

    Instance instance;

    instance.buildDefaultInterface();

    if (argc >= 2) {
        JST_CHECK_THROW(instance.openFlowgraphFile(argv[1]));
    }
    
    // Start compute thread.

    auto computeThread = std::thread([&]{
        while (instance.viewport().keepRunning()) {
            const auto result = instance.compute();

            if (result == Result::SUCCESS ||
                result == Result::TIMEOUT) {
                continue;
            }

            JST_CHECK_THROW(result);
        }
    });

    // Start graphical thread.

    auto graphicalThreadLoop = [](void* arg) {
        Instance* instance = reinterpret_cast<Instance*>(arg);

        if (instance->begin() == Result::SKIP) {
            return;
        }
        JST_CHECK_THROW(instance->present());
        if (instance->end() == Result::SKIP) {
            return;
        }
    };

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg(graphicalThreadLoop, &instance, 0, 1);
#else
    auto graphicalThread = std::thread([&]{
        while (instance.viewport().keepRunning()) {
            graphicalThreadLoop(&instance);
        }
    });
#endif

    // Start input polling.

#ifdef __EMSCRIPTEN__
    emscripten_runtime_keepalive_push();
#else
    while (instance.viewport().keepRunning()) {
        instance.viewport().pollEvents();
    }
#endif

    // Destruction.

    computeThread.join();
#ifndef __EMSCRIPTEN__
    graphicalThread.join();
#endif
    
    instance.destroy();

    std::cout << "Goodbye from CyberEther!" << std::endl;

    return 0;
}
