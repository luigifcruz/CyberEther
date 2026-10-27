#ifndef JETSTREAM_RENDER_SAKURA_DIV_HH
#define JETSTREAM_RENDER_SAKURA_DIV_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct Div : public Component {
    struct Config {
        Component* content = nullptr;
        Extent2D<F32> size = {Fill, 0.0f};
        F32 padding = 0.0f;
        F32 rounding = 0.0f;
        bool border = true;
        F32 borderWidth = 1.0f;
        bool chrome = false;
        bool shadow = false;
        bool waves = false;
        bool selected = false;
        std::string colorKey = "card";
        std::string hoveredColorKey = "header_hovered";
        std::string selectedColorKey = "header_hovered";
        std::string borderColorKey = "border";
        Callback<void()> onClick;
        Callback<void()> onDoubleClick;

        bool operator==(const Config&) const = default;
    };

    Div();
    ~Div();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DIV_HH
