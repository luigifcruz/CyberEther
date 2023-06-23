#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

constexpr static Device ComputeDevice = Device::CPU;
constexpr static Device RenderDevice  = Device::WebGPU;
using Platform = Viewport::GLFW<RenderDevice>;

class UI {
 public:
    UI(Instance& instance) : instance(instance) {

        JST_CHECK_THROW(instance.create());

        streaming = true;
        // graphicalWorker = std::thread([&]{ this->graphicalThreadLoop(); });
        // computeWorker = std::thread([&]{ this->computeThreadLoop(); });
    }

    ~UI() {
        streaming = false;
        // computeWorker.join();
        graphicalWorker.join();
        instance.destroy();

        JST_DEBUG("The UI was destructed.");
    }

 private:
    std::thread graphicalWorker;
    std::thread computeWorker;
    Instance& instance;
    bool streaming = false;

    // std::shared_ptr<Soapy<ComputeDevice>> sdr;
    // std::shared_ptr<Window<ComputeDevice>> win;
    // std::shared_ptr<Multiply<ComputeDevice>> win_mul;
    // std::shared_ptr<FFT<ComputeDevice>> fft;
    // std::shared_ptr<FFT<ComputeDevice>> ifft;
    // std::shared_ptr<Amplitude<ComputeDevice>> amp;
    // std::shared_ptr<Scale<ComputeDevice>> scl;
    // std::shared_ptr<Filter<ComputeDevice>> flt;
    // std::shared_ptr<Multiply<ComputeDevice>> flt_mul;

    // Bundle::LineplotUI<ComputeDevice> lpt;
    // Bundle::WaterfallUI<ComputeDevice> wtf;
    // Bundle::SpectrogramUI<ComputeDevice> spc;
    // Bundle::ConstellationUI<Device::CPU> cst;

    // void computeThreadLoop() {
    //     auto& buffer = sdr->getCircularBuffer();
    //     const auto& data = sdr->getOutputBuffer();

    //     while(streaming && instance.viewport().keepRunning()) {
    //         if (buffer.getOccupancy() < data.size()) {
    //             buffer.waitBufferOccupancy(data.size());
    //             continue;
    //         }

    //         if (instance.isCommited()) {
    //             JST_CHECK_THROW(instance.compute());
    //         }
    //     }
    // }

    void graphicalThreadLoop() {
        F32 stepSize = 10;
        // F32 frequency = sdr->getConfig().frequency / 1e6;

        while (streaming && instance.viewport().keepRunning()) {
            if (instance.begin() == Result::SKIP) {
                continue;
            }

            ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

            JST_CHECK_THROW(instance.present());
            JST_CHECK_THROW(instance.end());
        }

        JST_TRACE("UI Thread Safed");
    }
};

static void render_loop(void* instancePrototype) {
    Instance& instance = *(Instance*)instancePrototype;

    instance.viewport().pollEvents();

    if (instance.begin() == Result::SKIP) {
        return;
    }

    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

    JST_CHECK_THROW(instance.present());
    JST_CHECK_THROW(instance.end());
}

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;
    
    // Initialize Backends.
    if (Backend::Initialize<Device::CPU>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize CPU backend.");
        return 1;
    }

    if (Backend::Initialize<Device::WebGPU>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize WebGPU backend.");
        return 1;
    }

    // Initialize Instance.
    Instance instance;
    
    // Initialize Viewport.
    Viewport::Config viewportCfg;
    viewportCfg.vsync = true;
    viewportCfg.resizable = true;
    viewportCfg.size = {1512, 456};
    viewportCfg.title = "CyberEther";
    JST_CHECK_THROW(instance.buildViewport<Platform>(viewportCfg));

    // Initialize Window.
    Render::Window::Config renderCfg;
    renderCfg.imgui = true;
    JST_CHECK_THROW(instance.buildWindow<RenderDevice>(renderCfg));

    {
        // const Soapy<ComputeDevice>::Config& config {
        //     .deviceString = "driver=lime",
        //     .frequency = 96.9e6,
        //     .sampleRate = 10e6,
        //     .batchSize = 16,
        //     .outputBufferSize = 2 << 10,
        //     .bufferMultiplier = 512,
        // }; 
        auto ui = UI(instance);

        emscripten_set_main_loop_arg(render_loop, &instance, 0, 1);
    }

    std::cout << "Goodbye from CyberEther!" << std::endl;
}