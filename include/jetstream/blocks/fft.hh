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
        I64 axis = -1; // -1 means last axis

        JST_SERDES(forward, axis);
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
        return "Fast Fourier Transform that converts time-domain data to its frequency components.\n\n"
               "This block performs the FFT operation along a specified axis of the input tensor. "
               "The operation can be performed in either forward (time to frequency) or "
               "backward (frequency to time) direction.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor containing the data to transform\n\n"
               "Outputs:\n"
               "- buffer: Output tensor containing the transformed data\n\n"
               "Configuration:\n"
               "- Direction: Forward (time to frequency) or Backward (frequency to time)\n"
               "- Axis: The tensor axis along which to perform the FFT (default: last axis)\n\n"
               "Supported Data Types:\n"
               "- Complex Float32 to Complex Float32\n"
               "- Float32 to Complex Float32 (for real-to-complex transforms)";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            fft, "fft", {
                .forward = config.forward,
                .axis = config.axis,
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
        // FFT Direction
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
        
        // FFT Axis Selection
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Axis");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        
        // Get the tensor rank first (if available)
        I64 rank = 0;
        if (!input.buffer.empty()) {
            rank = static_cast<I64>(input.buffer.rank());
        }
        
        // Create a string representation of the current axis
        std::string axisLabel;
        if (config.axis < 0) {
            axisLabel = "Last axis";
        } else {
            axisLabel = "Axis " + std::to_string(config.axis);
        }
        
        // Display a dropdown to select the axis
        if (ImGui::BeginCombo("##fft-axis", axisLabel.c_str())) {
            // "Last axis" option
            if (ImGui::Selectable("Last axis", config.axis < 0)) {
                config.axis = -1;
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
            
            // Only show specific axis options if we have a valid input
            if (rank > 0) {
                for (I64 i = 0; i < rank; i++) {
                    std::string option = "Axis " + std::to_string(i);
                    if (ImGui::Selectable(option.c_str(), config.axis == i)) {
                        config.axis = i;
                        JST_DISPATCH_ASYNC([&](){
                            ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                            JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                        });
                    }
                }
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
