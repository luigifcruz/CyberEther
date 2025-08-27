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
        Extent2D<U64> viewSize = {512, 512};

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
        return "Visualizes modulated data in a 2D scatter plot, commonly used in digital communications to represent symbol modulation.\n\n"
               "The Constellation block creates a scatter plot visualization that plots the in-phase (I) component on the X-axis "
               "and the quadrature (Q) component on the Y-axis of complex data samples. This representation is crucial for "
               "analyzing digital modulation schemes such as QPSK, QAM, PSK, and other digital modulations.\n\n"
               "Inputs:\n"
               "- buffer: Complex-valued tensor (CF32 type) containing I/Q data samples.\n"
               "  - Each complex value becomes a point on the constellation diagram.\n"
               "  - For digital signals, these points tend to cluster around the expected symbol positions.\n\n"
               "Configuration Parameters:\n"
               "- viewSize: Size of the visualization in pixels (width, height)\n\n"
               "Visual Representation:\n"
               "- X-axis: In-phase (I) component of the complex signal\n"
               "- Y-axis: Quadrature (Q) component of the complex signal\n"
               "- Points: Individual signal samples visualized in I/Q space\n"
               "- Grid: Reference lines for zero crossings and unit circles\n\n"
               "Key Applications:\n"
               "- Analyzing digital modulation quality\n"
               "- Identifying modulation types\n"
               "- Diagnosing signal impairments (phase noise, amplitude imbalance, etc.)\n"
               "- Evaluating signal-to-noise ratio\n"
               "- Troubleshooting digital communication systems\n\n"
               "Interpretation:\n"
               "- Clear, distinct clusters indicate good signal quality\n"
               "- Smeared or scattered points suggest channel impairments or noise\n"
               "- Rotation indicates frequency offset or phase drift\n"
               "- Elliptical shape instead of circular suggests I/Q imbalance";
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
        const F32 width = (size.x < maxWidth) ? size.x : maxWidth;
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
