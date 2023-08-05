#ifndef JETSTREAM_BUNDLES_WATERFALL_BASE_HH
#define JETSTREAM_BUNDLES_WATERFALL_BASE_HH

#include "jetstream/bundle.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/waterfall.hh"

namespace Jetstream::Bundles {

template<Device D, typename T = F32>
class Waterfall : public Bundle {
 public:
   // Configuration 

    struct Config {
        F32 zoom = 1.0;
        I32 offset = 0;
        U64 height = 512;
        bool interpolate = true;
        Size2D<U64> viewSize = {4096, 512};

        JST_SERDES(
            JST_SERDES_VAL("zoom", zoom);
            JST_SERDES_VAL("offset", offset);
            JST_SERDES_VAL("height", height);
            JST_SERDES_VAL("interpolate", interpolate);
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
        return "waterfall-view";
    }

    constexpr std::string prettyName() const {
        return "Waterfall View";
    }

    // Constructor

    Waterfall(Instance& instance, const std::string& name, const Config& config, const Input& input)
         : config(config), input(input) {
        waterfall = instance.addModule<Jetstream::Waterfall, D, T>(name + "-ui", {
            .zoom = config.zoom,
            .offset = config.offset,
            .height = config.height,
            .interpolate = config.interpolate,
            .viewSize = config.viewSize,
        }, {
            .buffer = input.buffer,
        }, true);
    }
    virtual ~Waterfall() = default;

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
        auto interpolate = waterfall->interpolate();
        
        ImGui::Text("Interpolate");
        ImGui::SameLine(200);
        if (ImGui::Checkbox("##WaterfallInterpolate", &interpolate)) {
            waterfall->interpolate(interpolate);
        }

        ImGui::Text("Zoom");
        ImGui::SameLine(200);
        auto zoom = waterfall->zoom();
        if (ImGui::DragFloat("##WaterfallZoom", &zoom, 0.01, 1.0, 5.0, "%f", 0)) {
            waterfall->zoom(zoom);
        } 
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    Config config;
    Input input;
    Output output;

    std::shared_ptr<Jetstream::Waterfall<D, T>> waterfall;
    I32 position;

    ImVec2 getRelativeMousePos() {
        ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
        ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
        return ImVec2(mousePositionAbsolute.x - screenPositionAbsolute.x,
                      mousePositionAbsolute.y - screenPositionAbsolute.y);
    }
};
    
}  // namespace Jetstream::Bundles

#endif
