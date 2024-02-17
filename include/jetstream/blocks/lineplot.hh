#ifndef JETSTREAM_BLOCK_LINEPLOT_BASE_HH
#define JETSTREAM_BLOCK_LINEPLOT_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/lineplot.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Lineplot : public Block {
 public:
    // Configuration

    struct Config {
        U64 numberOfVerticalLines = 20;
        U64 numberOfHorizontalLines = 5;
        Size2D<U64> viewSize = {512, 384};
        F32 zoom = 1.0f;
        F32 translation = 0.0f;

        JST_SERDES(numberOfVerticalLines, numberOfHorizontalLines, viewSize, zoom, translation);
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
        return "lineplot";
    }

    std::string name() const {
        return "Lineplot";
    }

    std::string summary() const {
        return "Displays data in a line plot.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Visualizes input data in a line graph format, suitable for time-domain signals and waveform displays.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            lineplot, "lineplot", {
                .numberOfVerticalLines = config.numberOfVerticalLines,
                .numberOfHorizontalLines = config.numberOfHorizontalLines,
                .viewSize = config.viewSize,
                .zoom = config.zoom,
                .translation = config.translation
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(lineplot->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawPreview(const F32& maxWidth) {
        const auto& size = lineplot->viewSize();
        const auto& ratio = size.ratio();
        const F32 width = (size.width < maxWidth) ? size.width : maxWidth;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((maxWidth - width) / 2.0f));
        ImGui::Image(lineplot->getTexture().raw(), ImVec2(width, width/ratio));
    }

    constexpr bool shouldDrawPreview() const {
        return true;
    }

    void drawView() {
        auto [x, y] = ImGui::GetContentRegionAvail();
        auto scale = ImGui::GetIO().DisplayFramebufferScale;
        auto [width, height] = lineplot->viewSize({
            static_cast<U64>(x*scale.x),
            static_cast<U64>(y*scale.y)
        });
        ImGui::Image(lineplot->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));

        if (ImGui::IsItemHovered()) {
            // Handle zoom interaction.

            const auto& scroll = ImGui::GetIO().MouseWheel;    

            if (scroll != 0.0f) {
                auto [mouse_x, mouse_y] = GetRelativeMousePos();
                config.zoom += (1.0f / scroll);

                mouse_x *= scale.x;
                mouse_y *= scale.y;

                const auto& [zoom, translation] = lineplot->zoom({mouse_x, mouse_y}, config.zoom);

                config.zoom = zoom;
                config.translation = translation;
            }

            // Handle translation interaction.

            if (ImGui::IsAnyMouseDown()) {
                const auto& [dx, _] = ImGui::GetMouseDragDelta(0);

                if (dx != 0.0f) {
                    lineplot->translation((((dx * (1.0f / x)) * 2.0f) / config.zoom) + config.translation);
                }
            } else {
                config.translation = lineplot->translation();
            }
        }
    }

    constexpr bool shouldDrawView() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Lineplot<D, IT>> lineplot;

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Lineplot, is_specialized<Jetstream::Lineplot<D, IT>>::value &&
                           std::is_same<OT, void>::value)

#endif
