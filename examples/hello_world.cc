#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

// Selects the compute backend to use.
constexpr static Device ComputeDevice = Device::CPU;

class UI {
 public:
    UI(Instance& instance) : instance(instance) {
        JST_CHECK_THROW(create());
    }

    ~UI() {
        destroy();
    }

    Result create() {
        JST_CHECK(instance.addModule(
            sdr, "soapy", {
                .deviceString = "driver=rtlsdr",
                .frequency = 96.9e6,
                .sampleRate = 2.0e6,
                .numberOfBatches = 8,
                .numberOfTimeSamples = 2 << 10,
                .bufferMultiplier = 512,
            }, {}
        ));

        JST_CHECK(instance.addModule(
            win, "win", {
                .size = sdr->getOutputBuffer().shape()[1],
            }, {}
        ));

        JST_CHECK(instance.addModule(
            win_mul, "win_mul", {}, {
                .factorA = sdr->getOutputBuffer(),
                .factorB = win->getOutputWindow(),
            }
        ));

        JST_CHECK(instance.addModule(
            fft, "fft", {
                .forward = true,
            }, {
                .buffer = win_mul->getOutputProduct(),
            }
        ));

        JST_CHECK(instance.addModule(
            amp, "amp", {}, {
                .buffer = fft->getOutputBuffer(),
            }
        ));

        JST_CHECK(instance.addModule(
            scl, "scl", {
                .range = {-100.0, 0.0},
            }, {
                .buffer = amp->getOutputBuffer(),
            }
        ));

        JST_CHECK(instance.addModule(
            lpt, "lpt", {}, {
                .buffer = scl->getOutputBuffer(),
            }
        ));

        streaming = true;
        
        computeWorker = std::thread([&]{
            while(streaming && instance.viewport().keepRunning()) {
                JST_CHECK_THROW(instance.compute());
            }
        });

        graphicalWorker = std::thread([&]{
            while (streaming && instance.viewport().keepRunning()) {
                if (instance.begin() == Result::SKIP) {
                    continue;
                }

                ImGui::Begin("Lineplot");

                auto [x, y] = ImGui::GetContentRegionAvail();
                auto scale = ImGui::GetIO().DisplayFramebufferScale;
                auto [width, height] = lpt->viewSize({
                    static_cast<U64>(x*scale.x),
                    static_cast<U64>(y*scale.y)
                });
                ImGui::Image(lpt->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));

                ImGui::End();

                JST_CHECK_THROW(instance.present());
                if (instance.end() == Result::SKIP) {
                    continue;
                }
            }
        });

        return Result::SUCCESS;
    }

    Result destroy() {
        streaming = false;
        computeWorker.join();
        graphicalWorker.join();
        instance.destroy();
        Backend::DestroyAll();

        JST_DEBUG("The UI was destructed.");

        return Result::SUCCESS;
    }

 private:
    std::thread graphicalWorker;
    std::thread computeWorker;
    Instance& instance;
    bool streaming = false;

    std::shared_ptr<Soapy<ComputeDevice>> sdr;
    std::shared_ptr<Window<ComputeDevice>> win;
    std::shared_ptr<Multiply<ComputeDevice>> win_mul;
    std::shared_ptr<FFT<ComputeDevice>> fft;
    std::shared_ptr<Amplitude<ComputeDevice>> amp;
    std::shared_ptr<Scale<ComputeDevice>> scl;
    std::shared_ptr<Lineplot<ComputeDevice>> lpt;
};

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    // Initialize Instance.
    Instance instance;

    // Initialize Viewport and Window.
    JST_CHECK_THROW(instance.buildDefaultInterface());

    // Try to comment these line and see what happens!
    instance.compositor().showStore(false)
                         .showFlowgraph(true);

    {
        auto ui = UI(instance);

        while (instance.viewport().keepRunning()) {
            instance.viewport().pollEvents();
        }
    }

    std::cout << "Goodbye from CyberEther!" << std::endl;
}