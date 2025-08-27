#ifndef JETSTREAM_BLOCK_SPECTROGRAM_BASE_HH
#define JETSTREAM_BLOCK_SPECTROGRAM_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/spectrogram.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Spectrogram : public Block {
 public:
    // Configuration

    struct Config {
        U64 height = 256;
        Extent2D<U64> viewSize = {512, 384};

        JST_SERDES(height, viewSize);
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
        JST_SERDES();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "spectrogram";
    }

    std::string name() const {
        return "Spectrogram";
    }

    std::string summary() const {
        return "Displays a spectrogram of data.";
    }

    std::string description() const {
        return "Visualizes how frequencies of input data change over time, representing amplitude using color intensity.\n\n"
               "The Spectrogram block creates a 2D visualization similar to a waterfall plot, but optimized for real-time "
               "frequency analysis. It displays frequencies on the X-axis, time on the Y-axis, and uses color to represent "
               "signal amplitude at each time-frequency point.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor containing frequency domain data. Typically, this is the output of an FFT block.\n"
               "  - Each new data frame represents a single frequency slice at a specific point in time.\n"
               "  - The amplitude values in the data determine the color intensity.\n\n"
               "Configuration:\n"
               "- height: Maximum number of time slices to retain in history\n"
               "- viewSize: Size of the visualization in pixels (width, height)\n\n"
               "Visual Representation:\n"
               "- Horizontal axis (X): Frequency\n"
               "- Vertical axis (Y): Time (newest data at the bottom, scrolls upward)\n"
               "- Color: Signal intensity (typically cooler colors for lower amplitudes, warmer colors for higher amplitudes)\n\n"
               "Key Features:\n"
               "- Optimized for real-time spectral monitoring\n"
               "- Automatic scaling of color mapping to maximize visibility\n"
               "- Efficient memory usage with configurable history depth\n\n"
               "Differences from Waterfall:\n"
               "- Spectrogram is optimized for real-time display with simpler controls\n"
               "- Waterfall provides more visualization options and user interaction";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            spectrogram, "spectrogram", {
                .height = config.height,
                .viewSize = config.viewSize,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(spectrogram->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawPreview(const F32& maxWidth) {
        const auto& size = spectrogram->viewSize();
        const auto& ratio = size.ratio();
        const F32 width = (size.x < maxWidth) ? size.x : maxWidth;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((maxWidth - width) / 2.0f));
        ImGui::Image(spectrogram->getTexture().raw(), ImVec2(width, width/ratio));
    }

    constexpr bool shouldDrawPreview() const {
        return true;
    }

    void drawView() {
        auto [x, y] = ImGui::GetContentRegionAvail();
        auto scale = ImGui::GetIO().DisplayFramebufferScale;
        auto [width, height] = spectrogram->viewSize({
            static_cast<U64>(x*scale.x),
            static_cast<U64>(y*scale.y)
        });
        ImGui::Image(spectrogram->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));
    }

    constexpr bool shouldDrawView() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Spectrogram<D, IT>> spectrogram;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Spectrogram, is_specialized<Jetstream::Spectrogram<D, IT>>::value &&
                              std::is_same<OT, void>::value)

#endif
