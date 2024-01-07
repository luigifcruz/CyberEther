#ifndef JETSTREAM_BLOCK_WATERFALL_BASE_HH
#define JETSTREAM_BLOCK_WATERFALL_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/waterfall.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Waterfall : public Block {
 public:
    // Configuration

    struct Config {
        F32 zoom = 1.0;
        I32 offset = 0;
        U64 height = 512;
        bool interpolate = true;
        Size2D<U64> viewSize = {512, 384};

        JST_SERDES(zoom, offset, height, interpolate, viewSize);
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
        return "waterfall";
    }

    std::string name() const {
        return "Waterfall";
    }

    std::string summary() const {
        return "Displays data in a waterfall plot.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Visualizes frequency-domain data over time in a 2D color-coded format. Suitable for spectral analysis.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().template addModule<Jetstream::Waterfall, D, IT>(
            waterfall, "waterfall", {
                .zoom = config.zoom,
                .offset = config.offset,
                .height = config.height,
                .interpolate = config.interpolate,
                .viewSize = config.viewSize,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(waterfall->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawPreview(const F32& maxWidth) {
        const auto& size = waterfall->viewSize();
        const auto& ratio = size.ratio();
        const F32 width = (size.width < maxWidth) ? size.width : maxWidth;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((maxWidth - width) / 2.0f));
        ImGui::Image(waterfall->getTexture().raw(), ImVec2(width, width/ratio));
    }

    constexpr bool shouldDrawPreview() const {
        return true;
    }

    void drawView() {
        auto [x, y] = ImGui::GetContentRegionAvail();
        auto scale = ImGui::GetIO().DisplayFramebufferScale;
        auto [width, height] = waterfall->viewSize({
            static_cast<U64>(x*scale.x),
            static_cast<U64>(y*scale.y)
        });
        ImGui::Image(waterfall->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));

        if (ImGui::IsItemHovered() && ImGui::IsAnyMouseDown()) {
            if (position == 0) {
                position = (getRelativeMousePos().x / waterfall->zoom()) + waterfall->offset();
            }
            waterfall->offset(position - (getRelativeMousePos().x / waterfall->zoom()));
        } else {
            position = 0;
        }
    }

    constexpr bool shouldDrawView() const {
        return true;
    }

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Interpolate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        auto interpolate = waterfall->interpolate();
        if (ImGui::Checkbox("##Interpolate", &interpolate)) {
            waterfall->interpolate(interpolate);
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Zoom");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        auto zoom = waterfall->zoom();
        if (ImGui::DragFloat("##Zoom", &zoom, 0.01, 1.0, 5.0, "%f", 0)) {
            waterfall->zoom(zoom);
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Waterfall<D, IT>> waterfall;
    I32 position;

    ImVec2 getRelativeMousePos() {
        ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
        ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
        return ImVec2(mousePositionAbsolute.x - screenPositionAbsolute.x,
                      mousePositionAbsolute.y - screenPositionAbsolute.y);
    }

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Waterfall, is_specialized<Jetstream::Waterfall<D, IT>>::value &&
                            std::is_same<OT, void>::value)

#endif
