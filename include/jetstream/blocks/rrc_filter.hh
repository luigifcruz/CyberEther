#ifndef JETSTREAM_BLOCK_RRC_FILTER_BASE_HH
#define JETSTREAM_BLOCK_RRC_FILTER_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/rrc_filter.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class RRCFilter : public Block {
 public:
    // Configuration

    struct Config {
        F32 symbolRate = 1.0e6f;
        F32 sampleRate = 2.0e6f;
        F32 rollOff = 0.35f;
        U64 taps = 101;

        JST_SERDES(symbolRate, sampleRate, rollOff, taps);
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
        return "rrc-filter";
    }

    std::string name() const {
        return "RRC Filter";
    }

    std::string summary() const {
        return "Root raised cosine matched filter for PSK modulation.";
    }

    std::string description() const {
        return "The RRC Filter block implements a root raised cosine (RRC) matched filter optimized for "
               "PSK (Phase Shift Keying) modulation schemes including BPSK, QPSK, and 8PSK. This filter "
               "provides optimal signal-to-noise ratio performance for digital communications by matching "
               "the transmitter's pulse shaping filter.\n\n"

               "## Parameters:\n"
               "- **Sample Rate**: The sampling rate of the input signal in samples per second.\n"
               "- **Symbol Rate**: The symbol rate of the input signal in symbols per second.\n"
               "- **Roll-off Factor**: Controls the bandwidth and spectral efficiency trade-off (0.0 to 1.0).\n"
               "- **Taps**: Number of filter coefficients determining filter length and performance.\n\n"

               "## Useful For:\n"
               "- Matched filtering in PSK demodulation systems.\n"
               "- Maximizing signal-to-noise ratio in digital communication receivers.\n"
               "- Symbol timing recovery and synchronization applications.\n"
               "- Reducing intersymbol interference (ISI) in band-limited channels.\n\n"

               "## Examples:\n"
               "- QPSK demodulation:\n"
               "  Config: Symbol Rate=1MHz, Sample Rate=4MHz, Roll-off=0.35\n"
               "  Input: CF32[8192] → Output: CF32[8192]\n"
               "- BPSK with oversampling:\n"
               "  Config: Symbol Rate=500kHz, Sample Rate=2MHz, Roll-off=0.22\n"
               "  Input: CF32[4096] → Output: CF32[4096]\n\n"

               "## Implementation:\n"
               "Input → RRC Filter Module → Output\n"
               "1. RRC filter module generates optimal matched filter coefficients based on symbol rate and roll-off.\n"
               "2. Filter coefficients are computed using the standard RRC pulse shaping formula.\n"
               "3. Input signal is convolved with the RRC coefficients to maximize SNR for symbol detection.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            filter, "filter", {
                .symbolRate = config.symbolRate,
                .sampleRate = config.sampleRate,
                .rollOff = config.rollOff,
                .taps = config.taps,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, filter->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (filter) {
            JST_CHECK(instance().eraseModule(filter->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 sampleRate = config.sampleRate / JST_MHZ;
        if (ImGui::InputFloat("##rrc-sample-rate", &sampleRate, 1.0f, 1.0f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.sampleRate = sampleRate * JST_MHZ;
            JST_MODULE_UPDATE(filter, setSampleRate(config.sampleRate));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Symbol Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 symbolRate = config.symbolRate / JST_MHZ;
        if (ImGui::InputFloat("##rrc-symbol-rate", &symbolRate, 0.1f, 0.1f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.symbolRate = symbolRate * JST_MHZ;
            JST_MODULE_UPDATE(filter, setSymbolRate(config.symbolRate));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Roll-off Factor");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 rollOff = config.rollOff;
        if (ImGui::SliderFloat("##rrc-rolloff", &rollOff, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
            config.rollOff = rollOff;
            JST_MODULE_UPDATE(filter, setRollOff(config.rollOff));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Taps");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 taps = config.taps;
        if (ImGui::InputFloat("##rrc-taps", &taps, 2.0f, 2.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            U64 newTaps = static_cast<U64>(taps);
            config.taps = newTaps;
            JST_MODULE_UPDATE(filter, setTaps(newTaps));
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::RRCFilter<D, IT>> filter;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(RRCFilter, is_specialized<Jetstream::RRCFilter<D, IT>>::value &&
                            std::is_same<OT, void>::value)

#endif
