#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

// Selects the compute backend to use.
constexpr static Device ComputeDevice = Device::CPU;

// Selects the graphical backend to use.
#ifdef __EMSCRIPTEN__
constexpr static Device RenderDevice  = Device::WebGPU;
#else
constexpr static Device RenderDevice  = Device::Vulkan;
#endif

// Selects the viewport platform to use.
using Platform = Viewport::GLFW<RenderDevice>;

class UI {
 public:
    UI(Instance& instance) : instance(instance) {
        JST_CHECK_THROW(create());
    }

    ~UI() {
        destroy();
    }

    Result create() {
        JST_CHECK(instance.addModule<Soapy, ComputeDevice>(
            sdr, "soapy", {
                .deviceString = "driver=rtlsdr",
                .frequency = 96.9e6,
                .sampleRate = 2.0e6,
                .outputShape = {8, 2 << 10},
                .bufferMultiplier = 512,
            }, {}
        ));

        JST_CHECK(instance.addModule<Window, ComputeDevice>(
            win, "win", {
                .shape = sdr->getOutputBuffer().shape(),
            }, {}
        ));

        JST_CHECK(instance.addModule<Filter, ComputeDevice>(
            flt, "flt", {
                .signalSampleRate = sdr->getConfig().sampleRate,
                .filterSampleRate = 5e6,
                .filterCenter = -5e6,
                .shape = sdr->getOutputBuffer().shape(),
            }, {}
        ));

        JST_CHECK(instance.addModule<Multiply, ComputeDevice>(
            win_mul, "win_mul", {}, {
                .factorA = sdr->getOutputBuffer(),
                .factorB = win->getWindowBuffer(),
            }
        ));

        JST_CHECK(instance.addModule<FFT, ComputeDevice>(
            fft, "fft", {
                .forward = true,
            }, {
                .buffer = win_mul->getOutputProduct(),
            }
        ));

        JST_CHECK(instance.addModule<Multiply, ComputeDevice>(
            flt_mul, "flt_mul", {}, {
                .factorA = fft->getOutputBuffer(),
                .factorB = flt->getOutputCoeffs(),
            }
        ));

        JST_CHECK(instance.addModule<Amplitude, ComputeDevice>(
            amp, "amp", {}, {
                .buffer = flt_mul->getOutputProduct(),
            }
        ));

        JST_CHECK(instance.addModule<Scale, ComputeDevice>(
            scl, "scl", {
                .range = {-100.0, 0.0},
            }, {
                .buffer = amp->getOutputBuffer(),
            }
        ));

        JST_CHECK(instance.addModule<Bundles::Lineplot, ComputeDevice>(
            lpt, "lpt", {}, {
                .buffer = scl->getOutputBuffer(),
            }
        ));

        JST_CHECK(instance.addModule<Bundles::Waterfall, ComputeDevice>(
            wtf, "wtf", {}, {
                .buffer = scl->getOutputBuffer(),
            }
        ));

        JST_CHECK(instance.addModule<Bundles::Spectrogram, ComputeDevice>(
            spc, "spc", {}, {
                .buffer = scl->getOutputBuffer(),
            }
        ));

        streaming = true;
        frequency = sdr->getConfig().frequency / 1e6;
        
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

        return Result::SUCCESS;
    }

    Result destroy() {
        streaming = false;
        computeWorker.join();
#ifndef __EMSCRIPTEN__
        graphicalWorker.join();
#endif
        instance.destroy();

        JST_DEBUG("The UI was destructed.");

        return Result::SUCCESS;
    }

 private:
    std::thread graphicalWorker;
    std::thread computeWorker;
    Instance& instance;
    bool streaming = false;
    F32 stepSize = 10.0f;
    F32 frequency = 10.0f;

    std::shared_ptr<Soapy<ComputeDevice>> sdr;
    std::shared_ptr<Window<ComputeDevice>> win;
    std::shared_ptr<Multiply<ComputeDevice>> win_mul;
    std::shared_ptr<FFT<ComputeDevice>> fft;
    std::shared_ptr<Amplitude<ComputeDevice>> amp;
    std::shared_ptr<Scale<ComputeDevice>> scl;
    std::shared_ptr<Filter<ComputeDevice>> flt;
    std::shared_ptr<Multiply<ComputeDevice>> flt_mul;

    std::shared_ptr<Bundles::Lineplot<ComputeDevice>> lpt;
    std::shared_ptr<Bundles::Waterfall<ComputeDevice>> wtf;
    std::shared_ptr<Bundles::Spectrogram<ComputeDevice>> spc;

    void computeThreadLoop() {
        JST_CHECK_THROW(instance.compute());
    }

    static void callRenderLoop(void* ui_ptr) {
        reinterpret_cast<UI*>(ui_ptr)->graphicalThreadLoop();
    }

    void graphicalThreadLoop() {
        if (instance.begin() == Result::SKIP) {
            return;
        }

        {
            ImGui::Begin("FIR Filter Control");

            auto sampleRate = flt->filterSampleRate() / (1000 * 1000);
            if (ImGui::DragFloat("Bandwidth (MHz)", &sampleRate, 0.01, 0, sdr->getConfig().sampleRate / (1000 * 1000), "%.2f MHz")) {
                flt->filterSampleRate(sampleRate * (1000 * 1000));
            }

            auto center = flt->filterCenter() / (1000 * 1000);
            auto halfSampleRate = (sdr->getConfig().sampleRate / 2.0) / (1000 * 1000);
            if (ImGui::DragFloat("Center (MHz)", &center, 0.01, -halfSampleRate, halfSampleRate, "%.2f MHz")) {
                flt->filterCenter(center * (1000 * 1000));
            }

            auto numberOfTaps = static_cast<int>(flt->filterTaps());
            if (ImGui::DragInt("Taps", &numberOfTaps, 2, 3, 2000, "%d taps")) {
                flt->filterTaps(numberOfTaps);
            }

            ImGui::End();
        }

        {
            ImGui::Begin("Control");

            ImGui::InputFloat("Frequency (MHz)", &frequency, stepSize, stepSize, "%.3f MHz", ImGuiInputTextFlags_None);
            if (ImGui::IsItemEdited()) { frequency = sdr->setTunerFrequency(frequency * 1e6) / 1e6; }
            ImGui::InputFloat("Step Size (MHz)", &stepSize, 1.0f, 5.0f, "%.3f MHz");

            auto [min, max] = scl->range();
            if (ImGui::DragFloatRange2("Range (dBFS)", &min, &max,
                                       1, -300, 0, "Min: %.0f dBFS", "Max: %.0f dBFS")) {
                scl->range({min, max});
            }

            ImGui::End();
        }

        {
            ImGui::Begin("System Info");

            if (ImGui::CollapsingHeader("Buffer Health", ImGuiTreeNodeFlags_DefaultOpen)) {
                float bufferThroughputMB = (sdr->getCircularBuffer().getThroughput() / (1024 * 1024));
                ImGui::TextFormatted("Buffer Throughput {:.0f} MB/s", bufferThroughputMB);

                float bufferCapacityMB = ((F32)sdr->getCircularBuffer().getCapacity() * sizeof(CF32) / (1024 * 1024));
                ImGui::TextFormatted("Capacity {:.0f} MB", bufferCapacityMB);

                ImGui::TextFormatted("Overflows {}", sdr->getCircularBuffer().getOverflows());

                float bufferUsageRatio = (F32)sdr->getCircularBuffer().getOccupancy() /
                                                sdr->getCircularBuffer().getCapacity();
                ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
            }

            if (ImGui::CollapsingHeader("Compute", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextFormatted("Data Shape: ({}, {})", sdr->getConfig().outputShape[0], sdr->getConfig().outputShape[1]);
                ImGui::TextFormatted("Compute Device: {}", GetDevicePrettyName(ComputeDevice));
            }

            if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextFormatted("Viewport Device: {}", instance.viewport().name());
                ImGui::TextFormatted("Render Device: {}", GetDevicePrettyName(RenderDevice));
                ImGui::TextFormatted("Dropped Frames: {}", instance.window().stats().droppedFrames);
            }

            if (ImGui::CollapsingHeader("SDR", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextFormatted("Device Name: {}", sdr->getDeviceName());
                ImGui::TextFormatted("Hardware Key: {}", sdr->getDeviceHardwareKey());
                float sdrThroughputMB = ((sdr->getConfig().sampleRate * 8) / (1024 * 1024));
                ImGui::TextFormatted("Data Throughput {:.0f} MB/s", sdrThroughputMB);
                ImGui::TextFormatted("RF Bandwidth: {:.1f} MHz", sdr->getConfig().sampleRate / (1000 * 1000));
            }

            ImGui::End();
        }

        JST_CHECK_THROW(instance.present());
        if (instance.end() == Result::SKIP) {
            return;
        }
    }
};

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;
    
    // Initialize Backends.
    if (Backend::Initialize<ComputeDevice>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize compute backend.");
        return 1;
    }

    if (Backend::Initialize<RenderDevice>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize render backend.");
        return 1;
    }

    // Initialize Instance.
    Instance instance;
    
    // Initialize Viewport.
    Viewport::Config viewportCfg;
    viewportCfg.vsync = true;
    viewportCfg.size = {3130, 1140};
    viewportCfg.title = "CyberEther";
    JST_CHECK_THROW(instance.buildViewport<Platform>(viewportCfg));

    // Initialize Window.
    Render::Window::Config renderCfg;
    renderCfg.imgui = true;
    renderCfg.scale = 1.0;
    JST_CHECK_THROW(instance.buildRender<RenderDevice>(renderCfg));

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