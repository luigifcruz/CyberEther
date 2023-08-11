#ifndef JETSTREAM_BUNDLES_SOAPY_BASE_HH
#define JETSTREAM_BUNDLES_SOAPY_BASE_HH

#include "jetstream/bundle.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/soapy.hh"

namespace Jetstream::Bundles {

template<Device D, typename T = CF32>
class Soapy : public Bundle {
 public:
    // Configuration 

    struct Config {
        std::string deviceString;
        F32 frequency;
        F32 sampleRate;
        VectorShape<2> outputShape;
        U64 bufferMultiplier = 4;

        JST_SERDES(
            JST_SERDES_VAL("deviceString", deviceString);
            JST_SERDES_VAL("frequency", frequency);
            JST_SERDES_VAL("sampleRate", sampleRate);
            JST_SERDES_VAL("outputShape", outputShape);
            JST_SERDES_VAL("bufferMultiplier", bufferMultiplier);
        );
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
        Vector<D, T, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Vector<D, T, 2>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    constexpr std::string name() const {
        return "soapy-view";
    }

    constexpr std::string prettyName() const {
        return "Soapy Device View";
    }

    // Constructor

    Soapy(Instance& instance, const std::string& name, const Config& config, const Input& input)
         : config(config), input(input) {
        soapy = instance.addModule<Jetstream::Soapy, D, T>(name + "-ui", {
            .deviceString = config.deviceString,
            .frequency = config.frequency,
            .sampleRate = config.sampleRate,
            .outputShape = config.outputShape,
            .bufferMultiplier = config.bufferMultiplier,
        }, {}, true);

        output.buffer = soapy->getOutputBuffer();

        frequency = soapy->getConfig().frequency / 1e6;
    }
    virtual ~Soapy() = default;

    // Interface

    void drawInfo() {
        const auto& buffer = soapy->getCircularBuffer();

        ImGui::TextFormatted("Device Name: {}", soapy->getDeviceName());
        ImGui::TextFormatted("Hardware Key: {}", soapy->getDeviceHardwareKey());
        F32 sdrThroughputMB = ((soapy->getConfig().sampleRate * 8) / (1024 * 1024));
        ImGui::TextFormatted("Data Throughput {:.0f} MB/s", sdrThroughputMB);
        ImGui::TextFormatted("RF Bandwidth: {:.1f} MHz", soapy->getConfig().sampleRate / (1000 * 1000));

        ImGui::Separator();

        F32 bufferThroughputMB = (buffer.getThroughput() / (1024 * 1024));
        ImGui::TextFormatted("Buffer Throughput {:.0f} MB/s", bufferThroughputMB);

        F32 bufferCapacityMB = ((F32)buffer.getCapacity() * sizeof(CF32) / (1024 * 1024));
        ImGui::TextFormatted("Capacity {:.0f} MB", bufferCapacityMB);

        ImGui::TextFormatted("Overflows {}", buffer.getOverflows());

        F32 bufferUsageRatio = (F32)buffer.getOccupancy() / buffer.getCapacity();
        ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), "");
    }

    constexpr bool shouldDrawInfo() const {
        return true;
    }

    void drawControl() {
        ImGui::InputFloat("Frequency (MHz)", &frequency, stepSize, stepSize, "%.3f MHz", ImGuiInputTextFlags_None);
        if (ImGui::IsItemEdited()) { 
            frequency = soapy->setTunerFrequency(frequency * 1e6) / 1e6; 
        }
        ImGui::InputFloat("Step Size (MHz)", &stepSize, 1.0f, 5.0f, "%.3f MHz");    
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    Config config;
    Input input;
    Output output;

    F32 stepSize = 10.0f;
    F32 frequency = 10.0f;

    std::shared_ptr<Jetstream::Soapy<D, T>> soapy;
};

}  // namespace Jetstream::Bundles

#endif
