#ifndef JETSTREAM_BLOCK_FFT_BASE_HH
#define JETSTREAM_BLOCK_FFT_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/fft.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class FFT : public Block {
 public:
    // Configuration

    struct Config {
        bool forward = true;

        JST_SERDES(forward);
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
        return "fft";
    }

    std::string name() const {
        return "FFT";
    }

    std::string summary() const {
        return "Performs the fast fourier transform.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Fast Fourier Transform that converts time-domain data to its frequency components. Supports real and complex data types.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            fft, "fft", {
                .forward = config.forward,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, fft->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(fft->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Direction");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        static const char* forward = "Forward";
        static const char* backwards = "Backward";
        if (ImGui::BeginCombo("##fft-direction", config.forward ? forward : backwards)) {
            if (ImGui::Selectable(forward, config.forward)) {
                config.forward = true;
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
            if (config.forward) {
                ImGui::SetItemDefaultFocus();
            }

            if (ImGui::Selectable(backwards, !config.forward)) {
                config.forward = false;
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
            if (!config.forward) {
                ImGui::SetItemDefaultFocus();
            }

            ImGui::EndCombo();
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::FFT<D, IT, OT>> fft;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(FFT, is_specialized<Jetstream::FFT<D, IT, OT>>::value)

#endif
