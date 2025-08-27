#ifndef JETSTREAM_BLOCK_FM_BASE_HH
#define JETSTREAM_BLOCK_FM_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/fm.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class FM : public Block {
 public:
    // Configuration

    struct Config {
        F32 sampleRate = 240e3f;

        JST_SERDES(sampleRate);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, OT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, OT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "fm";
    }

    std::string name() const {
        return "FM";
    }

    std::string summary() const {
        return "Demodulates a FM signal.";
    }

    std::string description() const {
        return "Demodulates a complex-valued frequency modulated (FM) signal into its audio content.\n\n"
               "The FM block performs frequency demodulation on complex IQ samples (typically coming from an SDR or signal source), "
               "extracting the audio or data content that was encoded using frequency modulation. This block uses the "
               "polar discrimination method, which calculates the instantaneous phase difference between consecutive samples.\n\n"
               "Inputs:\n"
               "- buffer: Complex-valued input tensor (CF32 type) containing IQ samples of the FM signal.\n"
               "  - The signal should be centered at 0 Hz (baseband).\n"
               "  - For best results, the signal should be filtered to contain only the desired FM channel.\n\n"
               "Outputs:\n"
               "- buffer: Real-valued output tensor (F32 type) containing the demodulated audio signal.\n\n"
               "Technical Details:\n"
               "- Implements polar discriminator method for FM demodulation\n"
               "- Uses atan2(Q[n]*I[n-1] - I[n]*Q[n-1], I[n]*I[n-1] + Q[n]*Q[n-1]) for phase difference calculation\n"
               "- The output amplitude is proportional to the frequency deviation\n\n"
               "Applications:\n"
               "- FM broadcast radio reception\n"
               "- Amateur radio communications\n"
               "- Two-way radio demodulation\n"
               "- Wireless communications systems\n\n"
               "Tips for Use:\n"
               "- Input signal should be filtered to remove unwanted channels\n"
               "- For broadcast FM, typically follow with audio filtering and de-emphasis blocks\n"
               "- For narrowband FM, use appropriate bandwidth filtering before demodulation";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            fm, "fm", {
                .sampleRate = config.sampleRate,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, fm->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(fm->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 sampleRate = config.sampleRate / 1e6f;
        if (ImGui::InputFloat("##sample-rate", &sampleRate, 0.1f, 0.2f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.sampleRate = sampleRate * 1e6;

            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::FM<D, IT, OT>> fm;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(FM, is_specialized<Jetstream::FM<D, IT, OT>>::value &&
                     !std::is_same<OT, void>::value)

#endif
