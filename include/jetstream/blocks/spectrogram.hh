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
        Size2D<U64> viewSize = {512, 384};

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
        // TODO: Add decent block description describing internals and I/O.
        return "Visualizes how frequencies of input data change over time. Represents amplitude of frequencies using color.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().template addModule<Jetstream::Spectrogram, D, IT>(
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
        const F32 width = (size.width < maxWidth) ? size.width : maxWidth;
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

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Spectrogram, is_specialized<Jetstream::Spectrogram<D, IT>>::value &&
                              std::is_same<OT, void>::value)

#endif
