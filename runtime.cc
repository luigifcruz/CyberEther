#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

class UI {
 public:
    UI(Instance& instance) : instance(instance) {
        streaming = true;
        JST_CHECK_THROW(instance.create());

        // TODO: Abstract threading away.
        
        computeWorker = std::thread([&]{
            while(streaming && instance.viewport().keepRunning()) {
                this->computeThreadLoop();
            }
        });

#ifdef __EMSCRIPTEN__
        emscripten_set_main_loop_arg(callRenderLoop, this, 0, 1);
#else
        graphicalWorker = std::thread([&]{
            while (streaming && instance.viewport().keepRunning()) {
                this->graphicalThreadLoop(); 
            }
        });
#endif
    }

    ~UI() {
        streaming = false;
        computeWorker.join();
#ifndef __EMSCRIPTEN__
        graphicalWorker.join();
#endif  
        instance.destroy();

        JST_DEBUG("The UI was destructed.");
    }

 private:
    std::thread graphicalWorker;
    std::thread computeWorker;
    Instance& instance;

    bool streaming = false;

    void computeThreadLoop() {
        if (instance.isCommited()) {
            const auto result = instance.compute();

            if (result == Result::SUCCESS) {
                return;
            }
            
            if (result == Result::ERROR_TIMEOUT) {
                JST_WARN("Compute timeout occured.");
                return;
            }

            JST_CHECK_THROW(result);
        }
    }

    static void callRenderLoop(void* ui_ptr) {
        reinterpret_cast<UI*>(ui_ptr)->graphicalThreadLoop();
    }

    void graphicalThreadLoop() {
        if (instance.begin() == Result::SKIP) {
            return;
        }

        JST_CHECK_THROW(instance.present());
        if (instance.end() == Result::SKIP) {
            return;
        }
    }
};

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    // Initialize Instance.
    Instance instance;

    // Initialize Parser.
    Parser parser("/home/luigi/sandbox/CyberEther/multi-device.yml");
    parser.printAll();
    parser.importFromFile(instance);

    {
        auto ui = UI(instance);

#ifdef __EMSCRIPTEN__
        emscripten_runtime_keepalive_push();
#else
        while (instance.viewport().keepRunning()) {
            instance.viewport().pollEvents();
        }
#endif
    }

    std::cout << "Goodbye from CyberEther!" << std::endl;
}