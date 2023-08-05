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
        Size2D<U64> viewSize = {2048, 512};

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
        const Vector<D, T, 2> buffer;

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

    constexpr std::string name() const {
        return "spectrogram-view";
    }

    constexpr std::string prettyName() const {
        return "Spectrogram View";
    }

    // Constructor

    Spectrogram(Instance& instance, const std::string& name, const Config& config, const Input& input)
         : config(config), input(input) {
        spectrogram = instance.addModule<Jetstream::Spectrogram, D, T>(name + "-ui", {
            .height = config.height,
            .viewSize = config.viewSize,
        }, {
            .buffer = input.buffer,
        }, true);
    }
    virtual ~Spectrogram() = default;

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
    Config config;
    Input input;
    Output output;

    std::shared_ptr<Jetstream::Spectrogram<D, T>> spectrogram;
};

}  // namespace Jetstream::Bundles

#endif
