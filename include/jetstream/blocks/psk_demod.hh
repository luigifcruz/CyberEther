#ifndef JETSTREAM_BLOCK_PSK_DEMOD_BASE_HH
#define JETSTREAM_BLOCK_PSK_DEMOD_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/psk_demod.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class PskDemod : public Block {
 public:
    // Configuration

    struct Config {
        PskType pskType = PskType::QPSK;
        F64 sampleRate = 2000000.0;
        F64 symbolRate = 1000000.0;
        F64 frequencyLoopBandwidth = 0.05;
        F64 timingLoopBandwidth = 0.05;
        F64 dampingFactor = 0.707;
        F64 excessBandwidth = 0.35;
        U64 bufferSize = 8192;

        JST_SERDES(pskType, sampleRate, symbolRate, frequencyLoopBandwidth,
                   timingLoopBandwidth, dampingFactor, excessBandwidth, bufferSize);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        mem2::Tensor buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "psk_demod";
    }

    std::string name() const {
        return "PSK Demod";
    }

    std::string summary() const {
        return "Decodes PSK modulated signals with frequency and timing recovery.";
    }

    std::string description() const {
        return "The PSK Demod block demodulates PSK signals (BPSK, QPSK, 8-PSK) from I/Q data streams. "
               "It performs carrier frequency recovery using a Costas loop, symbol timing recovery using "
               "a Gardner detector, and outputs soft symbols. The block handles fine frequency "
               "correction and clock recovery automatically without requiring pre-filtering.\n\n"

               "## Arguments:\n"
               "- **PSK Type**: Modulation scheme - BPSK (2 symbols), QPSK (4 symbols), or 8-PSK (8 symbols).\n"
               "- **Sample Rate**: Input sample rate in Hz.\n"
               "- **Symbol Rate**: Expected symbol rate in Hz.\n"
               "- **Frequency Loop Bandwidth**: Carrier recovery loop bandwidth (0-1).\n"
               "- **Timing Loop Bandwidth**: Symbol timing recovery loop bandwidth (0-1).\n"
               "- **Damping Factor**: Loop filter damping coefficient for stability.\n"
               "- **Excess Bandwidth**: Root-raised cosine filter excess bandwidth (future use).\n"
               "- **Buffer Size**: Processing buffer size.\n\n"

               "## Useful For:\n"
               "- Demodulating digital satellite communication signals.\n"
               "- Recovering PSK data from software-defined radio streams.\n"
               "- Digital signal processing in communication systems.\n"
               "- Educational demonstrations of PSK demodulation.\n\n"

               "## Examples:\n"
               "- QPSK demodulation:\n"
               "  Config: PSK Type=QPSK, Sample Rate=2000000, Symbol Rate=500000\n"
               "  Input: CF32[8192] → Output: CF32[2048]\n"
               "- BPSK demodulation:\n"
               "  Config: PSK Type=BPSK, Sample Rate=1000000, Symbol Rate=125000\n"
               "  Input: CF32[8192] → Output: CF32[1024]\n\n"

               "## Implementation:\n"
               "Input → Frequency Correction → Timing Recovery → Soft Symbol Output\n"
               "1. Costas loop performs carrier phase and frequency tracking.\n"
               "2. Gardner timing error detector maintains symbol synchronization.\n"
               "3. Cubic interpolation provides fractional sample timing.\n"
               "4. Outputs frequency/phase corrected soft symbols without hard mapping.\n"
               "5. Soft symbols preserve amplitude and phase information for further decoding.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            pskDemod, "psk_demod", {
                .pskType = config.pskType,
                .sampleRate = config.sampleRate,
                .symbolRate = config.symbolRate,
                .frequencyLoopBandwidth = config.frequencyLoopBandwidth,
                .timingLoopBandwidth = config.timingLoopBandwidth,
                .dampingFactor = config.dampingFactor,
                .excessBandwidth = config.excessBandwidth,
                .bufferSize = config.bufferSize,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, pskDemod->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (pskDemod) {
            JST_CHECK(instance().eraseModule(pskDemod->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("PSK Type");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##PskType", config.pskType.string().c_str())) {
            for (const auto& [key, value] : config.pskType.rmap()) {
                bool isSelected = (config.pskType == key);
                if (ImGui::Selectable(value.c_str(), isSelected)) {
                    config.pskType = key;

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
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 sampleRate = config.sampleRate / JST_MHZ;
        if (ImGui::InputFloat("##sampleRate", &sampleRate, 0.1f, 0.1f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (sampleRate > 0) {
                config.sampleRate = sampleRate * JST_MHZ;

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Symbol Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 symbolRate = config.symbolRate / JST_MHZ;
        if (ImGui::InputFloat("##symbolRate", &symbolRate, 0.1f, 0.1f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (symbolRate > 0 && symbolRate < config.sampleRate) {
                config.symbolRate = symbolRate * JST_MHZ;

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Freq Loop");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 freqBW = static_cast<F32>(config.frequencyLoopBandwidth);
        if (ImGui::SliderFloat("##freqBW", &freqBW, 0.001f, 0.2f, "%.3f")) {
            config.frequencyLoopBandwidth = static_cast<F64>(freqBW);
            JST_MODULE_UPDATE(pskDemod, setFrequencyLoopBandwidth(config.frequencyLoopBandwidth));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Timing Loop");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 timingBW = static_cast<F32>(config.timingLoopBandwidth);
        if (ImGui::SliderFloat("##timingBW", &timingBW, 0.001f, 0.2f, "%.3f")) {
            config.timingLoopBandwidth = static_cast<F64>(timingBW);
            JST_MODULE_UPDATE(pskDemod, setTimingLoopBandwidth(config.timingLoopBandwidth));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Damping Factor");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 damping = static_cast<F32>(config.dampingFactor);
        if (ImGui::SliderFloat("##damping", &damping, 0.1f, 2.0f, "%.3f")) {
            config.dampingFactor = static_cast<F64>(damping);
            JST_MODULE_UPDATE(pskDemod, setDampingFactor(config.dampingFactor));
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::PskDemod<D, IT>> pskDemod;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(PskDemod, is_specialized<Jetstream::PskDemod<D, IT>>::value &&
                           std::is_same<OT, void>::value)

#endif
