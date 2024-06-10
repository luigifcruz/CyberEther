#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

// Selects the compute backend to use.
constexpr static Device ComputeDevice = Device::CPU;

// Selects the graphical backend to use.
constexpr static Device RenderDevice  = Device::Vulkan;

// Selects the viewport platform to use.
using ViewportPlatform = Viewport::GLFW<RenderDevice>;

class UI {
 public:
    UI(Instance& instance) : instance(instance) {}

    Result run() {
        // Add modules to the instance.

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

        JST_CHECK(instance.addModule(
            wtf, "wtf", {}, {
                .buffer = scl->getOutputBuffer(),
            }
        ));

        JST_CHECK(instance.addModule(
            spc, "spc", {}, {
                .buffer = scl->getOutputBuffer(),
            }
        ));

        // Start the instance.

        instance.start();

        computeWorker = std::thread([&]{
            while (instance.computing()) {
                this->computeThreadLoop();
            }
        });

        graphicalWorker = std::thread([&]{
            closing = false;
            frequency = sdr->getConfig().frequency / 1e6;

            while (instance.presenting()) {
                this->graphicalThreadLoop(); 
            }
        });

        // Wait user to close the window.

        while (instance.viewport().keepRunning()) {
            instance.viewport().pollEvents();
        }

        // Stop the instance and wait for threads.

        closing = true;
        instance.reset();
        instance.stop();

        if (computeWorker.joinable()) {
            computeWorker.join();
        }

        if (graphicalWorker.joinable()) {
            graphicalWorker.join();
        }

        // Destroy the instance.

        instance.destroy();

        return Result::SUCCESS;
    }

 private:
    std::thread graphicalWorker;
    std::thread computeWorker;
    Instance& instance;
    F32 stepSize = 10.0f;
    F32 frequency = 10.0f;
    bool closing;

    std::shared_ptr<Soapy<ComputeDevice>> sdr;
    std::shared_ptr<Window<ComputeDevice>> win;
    std::shared_ptr<Multiply<ComputeDevice>> win_mul;
    std::shared_ptr<FFT<ComputeDevice>> fft;
    std::shared_ptr<Amplitude<ComputeDevice>> amp;
    std::shared_ptr<Scale<ComputeDevice>> scl;

    std::shared_ptr<Lineplot<ComputeDevice>> lpt;
    std::shared_ptr<Waterfall<ComputeDevice>> wtf;
    std::shared_ptr<Spectrogram<ComputeDevice>> spc;

    void computeThreadLoop() {
        JST_CHECK_THROW(instance.compute());
    }

    void graphicalThreadLoop() {
        if (instance.begin() == Result::SKIP) {
            return;
        }

        if (!closing) {
            JST_CHECK_THROW(drawCustomViews());
        }

        JST_CHECK_THROW(instance.present());
        if (instance.end() == Result::SKIP) {
            return;
        }
    }

    Result drawCustomViews() {
        {
            ImGui::Begin("Control");

            float freq = frequency / 1e6;
            ImGui::InputFloat("Frequency (MHz)", &freq, stepSize, stepSize, "%.3f MHz", ImGuiInputTextFlags_None);
            frequency = freq * 1e6;
            if (ImGui::IsItemEdited()) { sdr->setTunerFrequency(freq); }
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
                float bufferThroughputMB = (sdr->getCircularBuffer().getThroughput() / JST_MB);
                ImGui::TextFormatted("Buffer Throughput {:.0f} MB/s", bufferThroughputMB);

                float bufferCapacityMB = ((F32)sdr->getCircularBuffer().getCapacity() * sizeof(CF32) / JST_MB);
                ImGui::TextFormatted("Capacity {:.0f} MB", bufferCapacityMB);

                ImGui::TextFormatted("Overflows {}", sdr->getCircularBuffer().getOverflows());

                float bufferUsageRatio = (F32)sdr->getCircularBuffer().getOccupancy() /
                                                sdr->getCircularBuffer().getCapacity();
                ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
            }

            if (ImGui::CollapsingHeader("Compute", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextFormatted("Data Shape: ({}, 1, {})", sdr->getConfig().numberOfBatches, sdr->getConfig().numberOfTimeSamples);
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
                float sdrThroughputMB = ((sdr->getConfig().sampleRate * 8) / JST_MB);
                ImGui::TextFormatted("Data Throughput {:.0f} MB/s", sdrThroughputMB);
                ImGui::TextFormatted("RF Bandwidth: {:.1f} MHz", sdr->getConfig().sampleRate / JST_MHZ);
            }

            ImGui::End();
        }

        {
            ImGui::Begin("Lineplot");

            auto [x, y] = ImGui::GetContentRegionAvail();
            auto scale = ImGui::GetIO().DisplayFramebufferScale;
            auto [width, height] = lpt->viewSize({
                static_cast<U64>(x*scale.x),
                static_cast<U64>(y*scale.y)
            });
            ImGui::Image(lpt->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));

            ImGui::End();
        }

        {
            ImGui::Begin("Waterfall");

            auto [x, y] = ImGui::GetContentRegionAvail();
            auto scale = ImGui::GetIO().DisplayFramebufferScale;
            auto [width, height] = wtf->viewSize({
                static_cast<U64>(x*scale.x),
                static_cast<U64>(y*scale.y)
            });
            ImGui::Image(wtf->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));

            ImGui::End();
        }

        {
            ImGui::Begin("Spectrogram");

            auto [x, y] = ImGui::GetContentRegionAvail();
            auto scale = ImGui::GetIO().DisplayFramebufferScale;
            auto [width, height] = spc->viewSize({
                static_cast<U64>(x*scale.x),
                static_cast<U64>(y*scale.y)
            });
            ImGui::Image(spc->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));

            ImGui::End();
        }

        return Result::SUCCESS;
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
    JST_CHECK_THROW(instance.buildViewport<ViewportPlatform>(viewportCfg));

    // Initialize Window.
    Render::Window::Config renderCfg;
    renderCfg.imgui = true;
    renderCfg.scale = 1.0;
    JST_CHECK_THROW(instance.buildRender<RenderDevice>(renderCfg));

    UI(instance).run();

    Backend::DestroyAll();

    std::cout << "Goodbye from CyberEther!" << std::endl;

    return 0;
}