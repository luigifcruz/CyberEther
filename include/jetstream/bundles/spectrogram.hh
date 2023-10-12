#ifndef JETSTREAM_BUNDLES_SPECTROGRAM_BASE_HH
#define JETSTREAM_BUNDLES_SPECTROGRAM_BASE_HH

#include "jetstream/bundle.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/spectrogram.hh"

namespace Jetstream::Bundles {

template<Device D, typename T = F32>
class Spectrogram : public Bundle {
 public:
    // Configuration

    struct Config {
        U64 height = 256;
        Size2D<U64> viewSize = {512, 384};

        JST_SERDES(
            JST_SERDES_VAL("height", height);
            JST_SERDES_VAL("viewSize", viewSize);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Vector<D, T, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
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

    std::string_view name() const {
        return "spectrogram-view";
    }

    std::string_view prettyName() const {
        return "Spectrogram";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule<Jetstream::Spectrogram, D, T>(
            spectrogram, "ui", {
                .height = config.height,
                .viewSize = config.viewSize,
            }, {
                .buffer = input.buffer,
            },
            locale().id
        ));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().removeModule("ui", locale().id));

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
    std::shared_ptr<Jetstream::Spectrogram<D, T>> spectrogram;

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Bundles

#endif
