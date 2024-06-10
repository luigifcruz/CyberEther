#ifndef JETSTREAM_BLOCK_CONSTELLATION_BASE_HH
#define JETSTREAM_BLOCK_CONSTELLATION_BASE_HH

#include "jetstream/parser.hh"
#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/constellation.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Constellation : public Block {
 public:
    // Configuration

    struct Config {
        Size2D<U64> viewSize = {512, 512};

        JST_SERDES(viewSize);
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
        return "constellation";
    }

    std::string name() const {
        return "Constellation";
    }

    std::string summary() const {
        return "Displays a constellation plot.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Visualizes modulated data in a 2D scatter plot. Commonly used in digital communication to represent symbol modulation.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            constellation, "constellation", {
                .viewSize = config.viewSize,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(constellation->locale()));

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

    void drawPreview(const F32& maxWidth) {
        const auto& size = constellation->viewSize();
        const auto& ratio = size.ratio();
        const F32 width = (size.width < maxWidth) ? size.width : maxWidth;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((maxWidth - width) / 2.0f));
        ImGui::Image(constellation->getTexture().raw(), ImVec2(width, width/ratio));
    }

    constexpr bool shouldDrawPreview() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Constellation<D, IT>> constellation;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Constellation, is_specialized<Jetstream::Constellation<D, IT>>::value &&
                                std::is_same<OT, void>::value)

#endif
