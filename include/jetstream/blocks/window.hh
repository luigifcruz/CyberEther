#ifndef JETSTREAM_BLOCK_WINDOW_BASE_HH
#define JETSTREAM_BLOCK_WINDOW_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/window.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Window : public Block {
 public:
    // Configuration

    struct Config {
        U64 size;

        JST_SERDES(size);
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
        Tensor<D, IT> window;

        JST_SERDES(window);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputWindow() const {
        return this->output.window;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "window";
    }

    std::string name() const {
        return "Window";
    }

    std::string summary() const {
        return "Generated a Butterworth window.";
    }

    std::string description() const {
        return "Generates a Butterworth window function of the specified length and order for signal processing applications.\n\n"
               "The Window block creates a Butterworth window function, which is a smoothly varying weighting function "
               "commonly applied to signals before frequency-domain analysis to reduce spectral leakage. The Butterworth "
               "window provides a good balance between main lobe width and side lobe suppression.\n\n"
               "Outputs:\n"
               "- buffer: Real-valued tensor (F32) containing the generated window coefficients.\n"
               "  - The tensor shape is [size], where size is the configured window length.\n"
               "  - Values range from 0 to 1, with higher values in the center of the window.\n\n"
               "Configuration Parameters:\n"
               "- size: Length of the window function in samples\n"
               "- order: Butterworth filter order that determines the window shape (higher order = steeper rolloff)\n\n"
               "Mathematical Properties:\n"
               "- Based on the Butterworth filter frequency response\n"
               "- Decreases smoothly and monotonically from center to edges\n"
               "- Higher order values create steeper transitions at the edges\n"
               "- The window is symmetric around its center\n\n"
               "Key Applications:\n"
               "- Pre-processing for FFT analysis\n"
               "- Reducing spectral leakage in frequency-domain operations\n"
               "- Smoothing signal transitions\n"
               "- Filter design\n"
               "- Time-domain windowing\n\n"
               "Usage Notes:\n"
               "- Typically used by multiplying it element-wise with a signal before FFT\n"
               "- Higher orders provide better frequency resolution but may attenuate more of the signal\n"
               "- The window size should match the size of the data segment being analyzed";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            window, "window", {
                .size = config.size,
            }, {},
            locale()
        ));

        JST_CHECK(Block::LinkOutput("window", output.window, window->getOutputWindow()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(window->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Taps");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 size = config.size;
        if (ImGui::InputFloat("##window-size", &size, 2.0f, 2.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.size = static_cast<U64>(size);

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
    std::shared_ptr<Jetstream::Window<D, IT>> window;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Window, is_specialized<Jetstream::Window<D, IT>>::value &&
                         std::is_same<OT, void>::value)

#endif
