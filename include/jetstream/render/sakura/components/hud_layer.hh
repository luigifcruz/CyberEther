#ifndef JETSTREAM_RENDER_SAKURA_HUD_LAYER_HH
#define JETSTREAM_RENDER_SAKURA_HUD_LAYER_HH

#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/components/overlay.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct HudLayer : public Layer {
    struct Hud {
        Component* content = nullptr;
        Anchor anchor = Anchor::TopLeft;
        F32 margin = 12.0f;
        Extent2D<F32> padding = {12.0f, 12.0f};
        F32 borderSize = 0.5f;
        F32 rounding = 12.0f;
        std::string backgroundColorKey = "background";
        std::string borderColorKey = "border";
        bool clickable = false;
        Callback<void()> onClick;
        Extent2D<F32> size = {0.0f, 0.0f};
        Extent2D<F32> offset = {0.0f, 0.0f};

        bool operator==(const Hud&) const = default;
    };

    struct Config {
        std::vector<Hud> huds;

        bool operator==(const Config&) const = default;
    };

    HudLayer();
    ~HudLayer();

    void update(Config config);
    bool covers(const Extent2D<F32>& point) const override;

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_HUD_LAYER_HH
