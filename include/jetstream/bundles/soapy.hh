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
        std::string streamString = "";
        F32 frequency;
        F32 sampleRate;
        VectorShape<2> outputShape;
        U64 bufferMultiplier = 4;

        JST_SERDES(
            JST_SERDES_VAL("deviceString", deviceString);
            JST_SERDES_VAL("streamString", streamString);
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

    std::string_view name() const {
        return "soapy-view";
    }

    std::string_view prettyName() const {
        return "Soapy";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance->addModule<Jetstream::Soapy, D, T>(
            soapy, "ui", {
                .deviceString = config.deviceString,
                .streamString = config.streamString,
                .frequency = config.frequency,
                .sampleRate = config.sampleRate,
                .outputShape = config.outputShape,
                .bufferMultiplier = config.bufferMultiplier,
            }, {},
            this->locale.id
        ));

        JST_CHECK(this->linkOutput("buffer", output.buffer, soapy->getOutputBuffer()));

        frequency = soapy->getConfig().frequency / 1e6;

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance->removeModule("ui", this->locale.id));

        return Result::SUCCESS;
    }

    // Interface

    void drawInfo() {
        const auto& buffer = soapy->getCircularBuffer();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Device Name:");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextFormatted("{} ({})", soapy->getDeviceName(), soapy->getDeviceHardwareKey());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Bandwidth:");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextFormatted("{:.1f} MHz", soapy->getConfig().sampleRate / (1000 * 1000));

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Overflows:");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextFormatted("{}", buffer.getOverflows());

        const F32& bufferOccupancy = buffer.getOccupancy();
        const F32 bufferOccupancyMB = (bufferOccupancy * sizeof(CF32) / (1024 * 1024));

        const F32& bufferCapacity = buffer.getCapacity();
        const F32 bufferCapacityMB = (bufferCapacity * sizeof(CF32) / (1024 * 1024));

        const F32& bufferThroughput = buffer.getThroughput();
        const F32 bufferThroughputMB = (bufferThroughput * sizeof(CF32) / (1024 * 1024));

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Buffer Health:");
        ImGui::TableSetColumnIndex(1);
        const F32 bufferUsageRatio = bufferOccupancy / bufferCapacity;
        const auto bufferOverlay = fmt::format("{:.0f}/{:.0f} MB", bufferOccupancyMB, bufferCapacityMB);
        ImGui::SetNextItemWidth(-1);
        ImGui::ProgressBar(bufferUsageRatio, ImVec2(0.0f, 0.0f), bufferOverlay.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Throughput:");
        ImGui::TableSetColumnIndex(1);
        const F32 sdrThroughputMB = ((soapy->getConfig().sampleRate * sizeof(CF32)) / (1024 * 1024));
        const F32 throughputRatio = (bufferThroughputMB / sdrThroughputMB) * 0.5f;
        const auto throughputOverlay = fmt::format("{:.0f}/{:.0f} MB/s", bufferThroughputMB, sdrThroughputMB);
        ImGui::SetNextItemWidth(-1);
        ImGui::ProgressBar(throughputRatio, ImVec2(0.0f, 0.0f), throughputOverlay.c_str());
    }

    constexpr bool shouldDrawInfo() const {
        return true;
    }

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Frequency (MHz)");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputFloat("##Frequency", &frequency, stepSize, stepSize, "%.3f MHz", ImGuiInputTextFlags_None);
        if (ImGui::IsItemEdited()) {
            frequency = soapy->setTunerFrequency(frequency * 1e6) / 1e6;
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Step Size (MHz)");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        ImGui::InputFloat("##StepSize", &stepSize, 1.0f, 5.0f, "%.3f MHz");
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    F32 stepSize = 10.0f;
    F32 frequency = 10.0f;

    std::shared_ptr<Jetstream::Soapy<D, T>> soapy;

    JST_DEFINE_BUNDLE_IO();
};

}  // namespace Jetstream::Bundles

#endif
