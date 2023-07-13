#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

constexpr static Device ComputeDevice = Device::CPU;
constexpr static Device RenderDevice  = Device::WebGPU;
using Platform = Viewport::GLFW<RenderDevice>;

class UI {
 public:
    UI(const Soapy<ComputeDevice>::Config& config, Instance& instance) : instance(instance) {
        sdr = instance.addBlock<Soapy, ComputeDevice>(config, {});

        win = instance.addBlock<Window, ComputeDevice>({
            .shape = sdr->getOutputBuffer().shape(),
        }, {});

        flt = instance.addBlock<Filter, ComputeDevice>({
            .signalSampleRate = sdr->getConfig().sampleRate,
            .filterSampleRate = 5e6,
            .filterCenter = -5e6,
            .shape = sdr->getOutputBuffer().shape(),
        }, {});

        win_mul = instance.addBlock<Multiply, ComputeDevice>({}, {
            .factorA = sdr->getOutputBuffer(),
            .factorB = win->getWindowBuffer(),
        });

        fft = instance.addBlock<FFT, ComputeDevice>({
            .direction = Direction::Forward,
        }, {
            .buffer = win_mul->getProductBuffer(),
        });

        flt_mul = instance.addBlock<Multiply, ComputeDevice>({}, {
            .factorA = fft->getOutputBuffer(),
            .factorB = flt->getCoeffsBuffer(),
        });

        amp = instance.addBlock<Amplitude, ComputeDevice>({}, {
            .buffer = flt_mul->getProductBuffer(),
        });

        scl = instance.addBlock<Scale, ComputeDevice>({
            .range = {-100.0, 0.0},
        }, {
            .buffer = amp->getOutputBuffer(),
        });

        ifft = instance.addBlock<FFT, ComputeDevice>({
            .direction = Direction::Backward,
        }, {
            .buffer = flt_mul->getProductBuffer(),
        });

        lpt.init(instance, {}, { .buffer = scl->getOutputBuffer(), });
        wtf.init(instance, {}, { .buffer = scl->getOutputBuffer(), });
        spc.init(instance, {}, { .buffer = scl->getOutputBuffer(), });
        cst.init(instance, {}, { .buffer = ifft->getOutputBuffer(), });

        JST_CHECK_THROW(instance.create());

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
    F32 stepSize = 10.0f;
    F32 frequency = 10.0f;

    std::shared_ptr<Soapy<ComputeDevice>> sdr;
    std::shared_ptr<Window<ComputeDevice>> win;
    std::shared_ptr<Multiply<ComputeDevice>> win_mul;
    std::shared_ptr<FFT<ComputeDevice>> fft;
    std::shared_ptr<FFT<ComputeDevice>> ifft;
    std::shared_ptr<Amplitude<ComputeDevice>> amp;
    std::shared_ptr<Scale<ComputeDevice>> scl;
    std::shared_ptr<Filter<ComputeDevice>> flt;
    std::shared_ptr<Multiply<ComputeDevice>> flt_mul;

    Bundle::LineplotUI<ComputeDevice> lpt;
    Bundle::WaterfallUI<ComputeDevice> wtf;
    Bundle::SpectrogramUI<ComputeDevice> spc;
    Bundle::ConstellationUI<Device::CPU> cst;

    void computeThreadLoop() {
        if (sdr->getCircularBuffer().getOccupancy() < sdr->getOutputBuffer().size()) {
            sdr->getCircularBuffer().waitBufferOccupancy(sdr->getOutputBuffer().size());
            return;
        }

        if (instance.isCommited()) {
            JST_CHECK_THROW(instance.compute());
        }
    }

    static void callRenderLoop(void* ui_ptr) {
        reinterpret_cast<UI*>(ui_ptr)->graphicalThreadLoop();
    }

    void graphicalThreadLoop() {
        if (instance.begin() == Result::SKIP) {
            return;
        }

        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        JST_CHECK_THROW(lpt.draw());
        JST_CHECK_THROW(wtf.draw());
        JST_CHECK_THROW(spc.draw());
        JST_CHECK_THROW(cst.draw());

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

            JST_CHECK_THROW(lpt.drawControl());
            JST_CHECK_THROW(wtf.drawControl());
            JST_CHECK_THROW(spc.drawControl());
            JST_CHECK_THROW(cst.drawControl());

            ImGui::End();
        }

        {
            ImGui::Begin("System Info");

            if (ImGui::CollapsingHeader("Buffer Health", ImGuiTreeNodeFlags_DefaultOpen)) {
                float bufferThroughputMB = (sdr->getCircularBuffer().getThroughput() / (1024 * 1024));
                ImGui::Text("Buffer Throughput %.0f MB/s", bufferThroughputMB);

                float bufferCapacityMB = ((F32)sdr->getCircularBuffer().getCapacity() * sizeof(CF32) / (1024 * 1024));
                ImGui::Text("Capacity %.0f MB", bufferCapacityMB);

                ImGui::Text("Overflows %llu", sdr->getCircularBuffer().getOverflows());

                float bufferUsageRatio = (F32)sdr->getCircularBuffer().getOccupancy() /
                                                sdr->getCircularBuffer().getCapacity();
                ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
            }

            if (ImGui::CollapsingHeader("Compute", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Data Shape: (%llu, %llu)", sdr->getConfig().batchSize, sdr->getConfig().outputBufferSize);
                ImGui::Text("Compute Device: %s", GetTypeName(ComputeDevice).c_str());
            }

            if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
                instance.getRender().drawDebugMessage();
                ImGui::Text("Viewport Device: %s", instance.getViewport().name().c_str());
                ImGui::Text("Render Device: %s", GetTypeName(RenderDevice).c_str());
                ImGui::Text("Dropped Frames: %llu", instance.window().stats().droppedFrames);
            }

            if (ImGui::CollapsingHeader("SDR", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Device Name: %s", sdr->getDeviceName().c_str());
                ImGui::Text("Hardware Key: %s", sdr->getDeviceHardwareKey().c_str());
                float sdrThroughputMB = ((sdr->getConfig().sampleRate * 8) / (1024 * 1024));
                ImGui::Text("Data Throughput %.0f MB/s", sdrThroughputMB);
                ImGui::Text("RF Bandwidth: %.1f MHz", sdr->getConfig().sampleRate / (1000 * 1000));
            }

            JST_CHECK_THROW(lpt.drawInfo());
            JST_CHECK_THROW(wtf.drawInfo());
            JST_CHECK_THROW(spc.drawInfo());
            JST_CHECK_THROW(cst.drawInfo());

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
    viewportCfg.resizable = true;
    viewportCfg.size = {3130, 1140};
    viewportCfg.title = "CyberEther";
    JST_CHECK_THROW(instance.buildViewport<Platform>(viewportCfg));

    // Initialize Window.
    Render::Window::Config renderCfg;
    renderCfg.imgui = true;
    renderCfg.scale = 1.0;
    JST_CHECK_THROW(instance.buildWindow<RenderDevice>(renderCfg));

    {
        const Soapy<ComputeDevice>::Config& config {
            .deviceString = "driver=airspy",
            .frequency = 96.9e6,
            .sampleRate = 2.5e6,
            .batchSize = 8,
            .outputBufferSize = 2 << 10,
            .bufferMultiplier = 512,
        }; 
        auto ui = UI(config, instance);

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