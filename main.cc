#include <thread>
#include <chrono>

#include "jetstream/base.hh"

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>

using namespace Jetstream;

using SystemViewport = Viewport::MacOS;
constexpr static Device ComputeDevice = Device::Metal;
constexpr static Device RenderDevice  = Device::Metal;

class SDR {
 public:
    struct Config {
        std::string deviceString;
        F64 frequency;
        F64 sampleRate;
        U64 batchSize;
        U64 outputBufferSize;
        U64 bufferMultiplier = 1024*8;
    };

    SDR(const Config& config, Instance& instance)
         : config(config), 
           data({config.batchSize, config.outputBufferSize}),
           buffer(config.outputBufferSize * config.bufferMultiplier) {
        streaming = true;
        deviceName = "None";
        deviceHardwareKey = "None";

        producer = std::thread([&]{ this->threadLoop(); });

        consumer = std::thread([&]{
            while (streaming) {
                if (buffer.getOccupancy() > data.size()) {
                    buffer.get(data.data(), data.size());
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

    constexpr const Vector<ComputeDevice, CF32, 2>& getOutputBuffer() const {
        return data;
    }

    constexpr const Memory::CircularBuffer<CF32>& getCircularBuffer() const {
        return buffer;
    }

    constexpr const Config& getConfig() const {
        return config;
    } 

    constexpr const std::string& getDeviceName() const {
        return deviceName;
    }

    constexpr const std::string& getDeviceHardwareKey() const {
        return deviceHardwareKey;
    }

    float setTunerFrequency(const F64& frequency) {
        SoapySDR::RangeList freqRange = device->getFrequencyRange(SOAPY_SDR_RX, 0);
        float minFreq = freqRange.front().minimum();
        float maxFreq = freqRange.back().maximum();

        config.frequency = frequency;

        if (frequency < minFreq) {
            config.frequency = minFreq;
        }

        if (frequency > maxFreq) {
            config.frequency = maxFreq;
        }

        device->setFrequency(SOAPY_SDR_RX, 0, config.frequency);

        return config.frequency;
    }

 private:
    std::thread producer;
    std::thread consumer;

    Config config;
    bool streaming = false;
    std::string deviceName;
    std::string deviceHardwareKey;
    Vector<ComputeDevice, CF32, 2> data;
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

            deviceName = device->getDriverKey();
            deviceHardwareKey = device->getHardwareKey();

            int flags;
            long long timeNs;
            CF32 tmp[8192];
            void *tmp_buffers[] = { tmp };

            streaming = true;
            while (streaming) {
                int ret = device->readStream(stream, tmp_buffers, 8192, flags, timeNs, 1e5);
                if (ret > 0) {
                    buffer.put(tmp, ret);
                }
            }

            device->deactivateStream(stream, 0, 0);
            device->closeStream(stream);
        }
    }
};

class UI {
 public:
    UI(SDR& sdr, Instance& instance)
             : sdr(sdr),
               instance(instance) {
        // Initialize Render
        Render::Window::Config renderCfg;
        renderCfg.imgui = true;
        JST_CHECK_THROW(instance.buildWindow<RenderDevice>(renderCfg));

        // Configure Jetstream
        win = instance.addBlock<Window, ComputeDevice>({
            .shape = sdr.getOutputBuffer().shape(),
        }, {});

        mul = instance.addBlock<Multiply, ComputeDevice>({}, {
            .factorA = sdr.getOutputBuffer(),
            .factorB = win->getWindowBuffer(),
        });

        fft = instance.addBlock<FFT, ComputeDevice>({}, {
            .buffer = mul->getProductBuffer(),
        });

        amp = instance.addBlock<Amplitude, ComputeDevice>({}, {
            .buffer = fft->getOutputBuffer(),
        });

        scl = instance.addBlock<Scale, ComputeDevice>({
            .range = {-100.0, 0.0},
        }, {
            .buffer = amp->getOutputBuffer(),
        });

        lpt.init(instance, {}, { .buffer = scl->getOutputBuffer(), });
        wtf.init(instance, {}, { .buffer = scl->getOutputBuffer(), });
        spc.init(instance, {}, { .buffer = scl->getOutputBuffer(), });

        JST_CHECK_THROW(instance.create());

        streaming = true;
        worker = std::thread([&]{ this->threadLoop(); });
    }

    ~UI() {
        streaming = false;
        worker.join();
        instance.destroy();

        JST_DEBUG("The UI was destructed.");
    }

 private:
    SDR& sdr;
    float frequency;
    std::thread worker;
    Instance& instance;
    bool streaming = false;

    std::shared_ptr<Window<ComputeDevice>> win;
    std::shared_ptr<Multiply<ComputeDevice>> mul;
    std::shared_ptr<FFT<ComputeDevice>> fft;
    std::shared_ptr<Amplitude<ComputeDevice>> amp;
    std::shared_ptr<Scale<ComputeDevice>> scl;

    Bundle::LineplotUI<ComputeDevice> lpt;
    Bundle::WaterfallUI<ComputeDevice> wtf;
    Bundle::SpectrogramUI<ComputeDevice> spc;

    void threadLoop() {
        frequency = sdr.getConfig().frequency / 1e6;

        F32 stepSize = 10;

        while (streaming && instance.viewport().keepRunning()) {
            if (instance.begin() == Result::SKIP) {
                continue;
            }

            ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

            JST_CHECK_THROW(lpt.draw());
            JST_CHECK_THROW(wtf.draw());
            JST_CHECK_THROW(spc.draw());

            {
                ImGui::Begin("Control");

                ImGui::InputFloat("Frequency (MHz)", &frequency, stepSize, stepSize, "%.3f MHz", ImGuiInputTextFlags_None);
                if (ImGui::IsItemEdited()) { frequency = sdr.setTunerFrequency(frequency * 1e6) / 1e6; }
                ImGui::InputFloat("Step Size (MHz)", &stepSize, 1.0f, 5.0f, "%.3f MHz");

                auto [min, max] = scl->range();
                if (ImGui::DragFloatRange2("dBFS Range", &min, &max,
                            1, -300, 0, "Min: %.0f dBFS", "Max: %.0f dBFS")) {
                    scl->range({min, max});
                }

                JST_CHECK_THROW(lpt.drawControl());
                JST_CHECK_THROW(wtf.drawControl());
                JST_CHECK_THROW(spc.drawControl());

                ImGui::End();
            }

            {
                ImGui::Begin("System Info");

                if (ImGui::CollapsingHeader("Buffer Health", ImGuiTreeNodeFlags_DefaultOpen)) {
                    float bufferThroughputMB = (sdr.getCircularBuffer().getThroughput() / (1024 * 1024));
                    ImGui::Text("Buffer Throughput %.0f MB/s", bufferThroughputMB);

                    float bufferCapacityMB = ((F32)sdr.getCircularBuffer().getCapacity() * sizeof(CF32) / (1024 * 1024));
                    ImGui::Text("Capacity %.0f MB", bufferCapacityMB);

                    ImGui::Text("Overflows %llu", sdr.getCircularBuffer().getOverflows());

                    float bufferUsageRatio = (F32)sdr.getCircularBuffer().getOccupancy() /
                                                  sdr.getCircularBuffer().getCapacity();
                    ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
                }

                if (ImGui::CollapsingHeader("Compute", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Text("Data Shape: (%lld, %lld)", sdr.getConfig().batchSize, sdr.getConfig().outputBufferSize);
                    ImGui::Text("Compute Device: %s", DeviceTypeInfo<ComputeDevice>().name);
                }

                if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
                    instance.getRender().drawDebugMessage();
                    ImGui::Text("Viewport Device: %s", instance.getViewport().name().c_str());
                    ImGui::Text("Render Device: %s", DeviceTypeInfo<RenderDevice>().name);
                    ImGui::Text("Dropped Frames: %lld", instance.window().stats().droppedFrames);
                }

                if (ImGui::CollapsingHeader("SDR", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Text("Device Name: %s", sdr.getDeviceName().c_str());
                    ImGui::Text("Hardware Key: %s", sdr.getDeviceHardwareKey().c_str());
                    float sdrThroughputMB = ((sdr.getConfig().sampleRate * 8) / (1024 * 1024));
                    ImGui::Text("Data Throughput %.0f MB/s", sdrThroughputMB);
                    ImGui::Text("RF Bandwidth: %.0f MHz", sdr.getConfig().sampleRate / (1000 * 1000));
                }

                JST_CHECK_THROW(lpt.drawInfo());
                JST_CHECK_THROW(wtf.drawInfo());
                JST_CHECK_THROW(spc.drawInfo());

                ImGui::End();
            }

            JST_CHECK_THROW(instance.present());
            JST_CHECK_THROW(instance.end());
        }
    }
};

int main() {
    std::cout << "Welcome to CyberEther!" << std::endl;
    
    // Initialize the backends.
    if (Backend::Initialize<Device::CPU>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize CPU backend.");
        return 1;
    }

    if (Backend::Initialize<Device::Metal>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize Metal backend.");
        return 1;
    }

    // Initialize Instance.
    Instance instance;
    
    // Initialize Viewport
    Viewport::Generic::Config viewportCfg;
    viewportCfg.vsync = true;
    viewportCfg.resizable = true;
    viewportCfg.size = {3130, 1140};
    viewportCfg.title = "CyberEther";
    JST_CHECK_THROW(instance.buildViewport<SystemViewport>(viewportCfg));

    {
        const SDR::Config& sdrConfig {
            .deviceString = "driver=lime",
            .frequency = 2.42e9,
            .sampleRate = 56e6,
            .batchSize = 16,
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
