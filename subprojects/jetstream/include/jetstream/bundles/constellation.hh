#ifndef JETSTREAM_BUNDLES_CONSTELLATION_BASE_HH
#define JETSTREAM_BUNDLES_CONSTELLATION_BASE_HH

#include "jetstream/parser.hh"
#include "jetstream/bundle.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/constellation.hh"

namespace Jetstream::Bundles {

template<Device D, typename T = CF32>
class Constellation : public Bundle {
 public:
    // Configuration

    struct Config {
        Size2D<U64> viewSize = {512, 512};

        JST_SERDES(
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

    constexpr std::string name() const {
        return "constellation-view";
    }

    constexpr std::string prettyName() const {
        return "Constellation";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance->addModule<Jetstream::Constellation, D, T>(
            constellation, "ui", {
                .viewSize = config.viewSize,
            }, {
                .buffer = input.buffer,
            },
            this->locale.id
        ));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance->removeModule("ui", this->locale.id));

        return Result::SUCCESS;
    }

    // Interface

    void drawView() {
        auto [x, y] = ImGui::GetContentRegionAvail();
        auto scale = ImGui::GetIO().DisplayFramebufferScale;
        auto [width, height] = constellation->viewSize({
            static_cast<U64>(x*scale.x),
            static_cast<U64>(y*scale.y)
        });
        ImGui::Image(constellation->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));
    }

    constexpr bool shouldDrawView() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Constellation<D, T>> constellation;

    JST_DEFINE_BUNDLE_IO();
};

}  // namespace Jetstream::Bundles

#endif
