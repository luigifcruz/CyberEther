#ifndef JETSTREAM_BLOCK_SOAPY_BASE_HH
#define JETSTREAM_BLOCK_SOAPY_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/soapy.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Soapy : public Block {
 public:
    using SoapyModule = Jetstream::Soapy<D, IT>;

    // Configuration

    struct Config {
        std::string hintString = "";
        std::string deviceString = "";
        std::string streamString = "";
        F32 frequency = 96.9e6;
        F32 sampleRate = 2.0e6;
        bool automaticGain = true;
        U64 numberOfBatches = 8;
        U64 numberOfTimeSamples = 8192;
        U64 bufferMultiplier = 4;

        JST_SERDES(hintString, deviceString, streamString,
                   frequency, sampleRate, automaticGain, 
                   numberOfBatches, numberOfTimeSamples, 
                   bufferMultiplier);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "soapy";
    }

    std::string name() const {
        return "Soapy";
    }

    std::string summary() const {
        return "Interface for SoapySDR devices.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Provides an interface to communicate and control SoapySDR supported devices, facilitating data acquisition and device configuration.";
    }

    // Constructor

    Result create() {
        // Preload configuration device string.
        std::string deviceString = config.deviceString;

        // Gather list of available devices according to the hint string.
        availableDeviceList = SoapyModule::ListAvailableDevices(config.hintString);

        // Load the first device if device string is empty and there are devices available.
        if (deviceString.empty() && !availableDeviceList.empty()) {
            const auto& [_, device] = *availableDeviceList.begin();
            deviceString = device.toString();
        }

        // Starting sub-modules.

        JST_CHECK(instance().addModule(
            soapy, "soapy", {
                .deviceString = deviceString,
                .streamString = config.streamString,
                .frequency = config.frequency,
                .sampleRate = config.sampleRate,
                .automaticGain = config.automaticGain,
                .numberOfBatches = config.numberOfBatches,
                .numberOfTimeSamples = config.numberOfTimeSamples,
                .bufferMultiplier = config.bufferMultiplier,
            }, {},
            locale()
        ));

        // Connecting sub-modules outputs.

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, soapy->getOutputBuffer()));

        // Fetching configuration.

        currentDevice = soapy->getDeviceLabel();

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(soapy->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawInfo() {
        const auto& buffer = soapy->getCircularBuffer();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Device Name");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextFormatted("{} ({})", soapy->getDeviceName(), soapy->getDeviceHardwareKey());

        const F32& bufferOccupancy = buffer.getOccupancy();
        const F32 bufferOccupancyMB = (bufferOccupancy * sizeof(CF32) / JST_MB);

        const F32& bufferCapacity = buffer.getCapacity();
        const F32 bufferCapacityMB = (bufferCapacity * sizeof(CF32) / JST_MB);

        const F32& bufferThroughput = buffer.getThroughput();
        const F32 bufferThroughputMB = (bufferThroughput * sizeof(CF32) / JST_MB);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Buffer Health");
        ImGui::TableSetColumnIndex(1);
        const F32 bufferUsageRatio = bufferOccupancy / bufferCapacity;
        const auto bufferOverlay = jst::fmt::format("{:.0f}/{:.0f} MB ({})", bufferOccupancyMB, bufferCapacityMB, buffer.getOverflows());
        ImGui::SetNextItemWidth(-1);
        ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), bufferOverlay.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Throughput");
        ImGui::TableSetColumnIndex(1);
        const F32 sdrThroughputMB = ((soapy->getConfig().sampleRate * sizeof(CF32)) / JST_MB);
        const F32 throughputRatio = (bufferThroughputMB / sdrThroughputMB) * 0.5f;
        const auto throughputOverlay = jst::fmt::format("{:.0f}/{:.0f} MB/s", bufferThroughputMB, sdrThroughputMB);
        ImGui::SetNextItemWidth(-1);
        ImGui::ProgressBar(throughputRatio, ImVec2(0.0f, 0.0f), throughputOverlay.c_str());
    }

    constexpr bool shouldDrawInfo() const {
        return true;
    }

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Batch / Samples");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        I32 taps[2] = {
            static_cast<I32>(config.numberOfBatches), 
            static_cast<I32>(config.numberOfTimeSamples)
        };
        if (ImGui::InputInt2("##batch-samples", taps, ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.numberOfBatches = static_cast<U64>(taps[0]);
            config.numberOfTimeSamples = static_cast<U64>(taps[1]);

            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 sampleRate = config.sampleRate / 1e6f;
        if (ImGui::InputFloat("##SampleRate", &sampleRate, 1.0f, 2.0f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.sampleRate = sampleRate * 1e6;
            JST_MODULE_UPDATE(soapy, setSampleRate(config.sampleRate));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Automatic Gain");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Checkbox("##AutomaticGain", &config.automaticGain)) {
            JST_MODULE_UPDATE(soapy, setAutomaticGain(config.automaticGain));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Device Hint");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##DeviceHintInput", &config.hintString);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Device List");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        static const char* noDeviceMessage = "No device found";
        if (ImGui::BeginCombo("##DeviceList", availableDeviceList.empty() ? noDeviceMessage : currentDevice.c_str())) {
            for (const auto& [label, device] : availableDeviceList) {
                bool isSelected = (currentDevice == label);
                if (ImGui::Selectable(label.c_str(), isSelected)) {
                    currentDevice = label;
                    config.deviceString = device.toString();

                    JST_DISPATCH_ASYNC([&](){
                        ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                        JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                    });
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1);
        const F32 fullWidth = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button("Reload Device List", ImVec2(fullWidth, 0))) {
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading device list..." });
                availableDeviceList = SoapyModule::ListAvailableDevices(config.hintString);
                JST_CHECK_NOTIFY(Result::SUCCESS);
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Frequency");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 frequency = config.frequency / JST_MHZ;
        if (ImGui::InputFloat("##Frequency", &frequency, stepSize, stepSize, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.frequency = frequency * JST_MHZ;
            JST_MODULE_UPDATE(soapy, setTunerFrequency(config.frequency));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Step Size");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputFloat("##StepSize", &stepSize, 1.0f, 5.0f, "%.3f MHz");
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    F32 stepSize = 10.0f;
    std::string currentDevice;
    typename SoapyModule::DeviceList availableDeviceList;
    
    std::shared_ptr<SoapyModule> soapy;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Soapy, is_specialized<Jetstream::Soapy<D, IT>>::value &&
                        std::is_same<OT, void>::value)

#endif
