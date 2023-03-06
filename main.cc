#include <thread>
#include <chrono>

#include "jetstream/base.hh"

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>

using namespace Jetstream;

class SDR {
 public:
    struct Config {
        std::string deviceString;
        F64 frequency;
        F64 sampleRate;
        U64 outputBufferSize;
        U64 bufferMultiplier = 1024*8;
    };

    SDR(const Config& config, Instance& instance)
         : config(config), 
           data(config.outputBufferSize),
           buffer(config.outputBufferSize * config.bufferMultiplier) {
        streaming = true;

        producer = std::thread([&]{ this->threadLoop(); });

        consumer = std::thread([&]{
            while (streaming) {
                if (buffer.GetOccupancy() > config.outputBufferSize) {
                    buffer.Get(data.data(), config.outputBufferSize);
                    instance.compute();
                }
            }
        });
    }

    ~SDR() {
        streaming = false;
        consumer.join();
        producer.join();
    }

    constexpr const Vector<Device::CPU, CF32>& getOutputBuffer() const {
        return data;
    }

    constexpr const Memory::CircularBuffer<CF32>& getCircularBuffer() const {
        return buffer;
    }

    constexpr const Config& getConfig() const {
        return config;
    } 

    void setTunerFrequency(const F64& frequency) {
        config.frequency = frequency;
        device->setFrequency(SOAPY_SDR_RX, 0, config.frequency);
    }

 private:
    std::thread producer;
    std::thread consumer;

    Config config;
    bool streaming = false;
    Vector<Device::CPU, CF32> data;
    Memory::CircularBuffer<CF32> buffer;

    SoapySDR::Device* device;
    SoapySDR::Stream* stream;

    void threadLoop() {
        while (streaming) {
            device = SoapySDR::Device::make(config.deviceString);

            if (device == nullptr) {
                JST_FATAL("Can't open device.");
                JST_CHECK_THROW(Result::ERROR);
            }

            device->setSampleRate( SOAPY_SDR_RX, 0, config.sampleRate);
            device->setFrequency(SOAPY_SDR_RX, 0, config.frequency);
            device->setGainMode(SOAPY_SDR_RX, 0, true);
            
            stream = device->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
            if (stream == nullptr) {
                JST_FATAL("Failed to setup stream.");
                SoapySDR::Device::unmake(device);
                JST_CHECK_THROW(Result::ERROR);
            }
            device->activateStream(stream, 0, 0, 0);

            int flags;
            long long timeNs;
            CF32 tmp[8192];
            void *tmp_buffers[] = { tmp };

            streaming = true;
            while (streaming) {
                int ret = device->readStream(stream, tmp_buffers, 8192, flags, timeNs, 1e5);
                if (ret > 0) {
                    buffer.Put(tmp, ret);
                }
            }

            device->deactivateStream(stream, 0, 0);
            device->closeStream(stream);
        }
    }
};

class UI {
 public:
    UI(SDR& sdr, Instance& instance) : sdr(sdr), instance(instance) {
        // Initialize Render
        Render::Window::Config renderCfg;
        renderCfg.imgui = true;
        JST_CHECK_THROW(instance.buildWindow<Device::Metal>(renderCfg));

        // Configure Jetstream
        win = instance.addBlock<Window, Device::CPU>({
            .size = sdr.getOutputBuffer().size(),
        }, {});

        mul = instance.addBlock<Multiply, Device::CPU>({
            .size = sdr.getOutputBuffer().size(),
        }, {
            .factorA = sdr.getOutputBuffer(),
            .factorB = win->getWindowBuffer(),
        });

        fft = instance.addBlock<FFT, Device::CPU>({
            .size = sdr.getOutputBuffer().size(),
        }, {
            .buffer = mul->getProductBuffer(),
        });

        amp = instance.addBlock<Amplitude, Device::CPU>({
            .size = sdr.getOutputBuffer().size(),
        }, {
            .buffer = fft->getOutputBuffer(),
        });

        scl = instance.addBlock<Scale, Device::CPU>({
            .size = sdr.getOutputBuffer().size(),
            .range = {-100.0, 0.0},
        }, {
            .buffer = amp->getOutputBuffer(),
        });

        lpt = instance.addBlock<Lineplot, Device::CPU>({}, {
            .buffer = scl->getOutputBuffer(),
        });

        wtf = instance.addBlock<Waterfall, Device::CPU>({}, {
            .buffer = scl->getOutputBuffer(),
        });

        spc = instance.addBlock<Spectrogram, Device::CPU>({}, {
            .buffer = scl->getOutputBuffer(),
        });

        JST_CHECK_THROW(instance.commit());

        streaming = true;
        worker = std::thread([&]{ this->threadLoop(); });
    }

    ~UI() {
        streaming = false;
        worker.join();

        JST_DEBUG("The UI was destructed.");
    }

 private:
    SDR& sdr;
    float frequency;
    std::thread worker;
    Instance& instance;
    bool streaming = false;

    std::shared_ptr<Window<Device::CPU>> win;
    std::shared_ptr<Multiply<Device::CPU>> mul;
    std::shared_ptr<FFT<Device::CPU>> fft;
    std::shared_ptr<Amplitude<Device::CPU>> amp;
    std::shared_ptr<Scale<Device::CPU>> scl;
    std::shared_ptr<Lineplot<Device::CPU>> lpt;
    std::shared_ptr<Waterfall<Device::CPU>> wtf;
    std::shared_ptr<Spectrogram<Device::CPU>> spc;

    ImVec2 getRelativeMousePos() {
        ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
        ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
        return ImVec2(mousePositionAbsolute.x - screenPositionAbsolute.x,
                      mousePositionAbsolute.y - screenPositionAbsolute.y);
    }

    void threadLoop() {
        int position;

        frequency = sdr.getConfig().frequency;

        while (streaming && instance.viewport().keepRunning()) {
            if (instance.begin() == Result::SKIP) {
                continue;
            }

            ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

            {
                ImGui::Begin("Waterfall");

                auto [x, y] = ImGui::GetContentRegionAvail();
                auto [width, height] = wtf->viewSize({(U64)x, (U64)y});
                ImGui::Image(wtf->getTexture().raw(), ImVec2(width, height));

                if (ImGui::IsItemHovered() && ImGui::IsAnyMouseDown()) {
                    if (position == 0) {
                        position = (getRelativeMousePos().x / wtf->zoom()) + wtf->offset();
                    }
                    wtf->offset(position - (getRelativeMousePos().x / wtf->zoom()));
                } else {
                    position = 0;
                }

                ImGui::End();
            }

            {
                ImGui::Begin("Spectrogram");

                auto [x, y] = ImGui::GetContentRegionAvail();
                auto [width, height] = spc->viewSize({(U64)x, (U64)y});
                ImGui::Image(spc->getTexture().raw(), ImVec2(width, height));

                ImGui::End();
            }

            {
                ImGui::Begin("Lineplot");
                
                auto [x, y] = ImGui::GetContentRegionAvail();
                auto [width, height] = lpt->viewSize({(U64)x, (U64)y});
                ImGui::Image(lpt->getTexture().raw(), ImVec2(width, height));

                ImGui::End();
            }

            {
                ImGui::Begin("Control");

                ImGui::InputFloat("Frequency (Hz)", &frequency);
                if (ImGui::Button("Tune")) {
                    sdr.setTunerFrequency(frequency);
                }

                auto [min, max] = scl->range();
                if (ImGui::DragFloatRange2("dBFS Range", &min, &max,
                            1, -300, 0, "Min: %.0f dBFS", "Max: %.0f dBFS")) {
                    scl->range({min, max});
                }

                auto interpolate = wtf->interpolate();
                if (ImGui::Checkbox("Interpolate Waterfall", &interpolate)) {
                    wtf->interpolate(interpolate);
                }

                auto zoom = wtf->zoom();
                if (ImGui::DragFloat("Waterfall Zoom", &zoom, 0.01, 1.0, 5.0, "%f", 0)) {
                    wtf->zoom(zoom);
                }

                ImGui::End();
            }

            {
                ImGui::Begin("Buffer Info");

                float bufferThroughputMB = (sdr.getCircularBuffer().GetThroughput() / (1024 * 1024));
                ImGui::Text("Throughput %.0f MB/s", bufferThroughputMB);

                float bufferCapacityMB = ((F32)sdr.getCircularBuffer().GetCapacity() * sizeof(CF32) / (1024 * 1024));
                ImGui::Text("Capacity %.0f MB", bufferCapacityMB);

                ImGui::Text("Overflows %llu", sdr.getCircularBuffer().GetOverflows());
                ImGui::Text("Dropped Frames: %lld", instance.window().stats().droppedFrames);

                ImGui::Separator();
                ImGui::Spacing();

                float bufferUsageRatio = (F32)sdr.getCircularBuffer().GetOccupancy() /
                                              sdr.getCircularBuffer().GetCapacity();
                ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
                ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
                ImGui::Text("Usage");

                ImGui::End();
            }

            JST_CHECK_THROW(instance.present());
            JST_CHECK_THROW(instance.end());
        }
    }
};

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;

    // Initialize Instance.
    Instance instance;
    
    // Initialize Viewport
    Viewport::Generic::Config viewportCfg;
    viewportCfg.vsync = true;
    viewportCfg.resizable = true;
    viewportCfg.size = {3130, 1140};
    viewportCfg.title = "CyberEther";
    JST_CHECK_THROW(instance.buildViewport<Viewport::MacOS>(viewportCfg));

    {
        const SDR::Config& sdrConfig {
            .deviceString = "driver=lime",
            .frequency = 2.42e9,
            .sampleRate = 30e6,
            .outputBufferSize = 2 << 10,
        }; 
        auto sdr = SDR(sdrConfig, instance);
        auto ui = UI(sdr, instance);

        while (instance.viewport().keepRunning()) {
            instance.viewport().pollEvents();
        }
    }

    std::cout << "Goodbye from CyberEther!" << std::endl;
}
