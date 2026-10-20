#ifndef JETSTREAM_RENDER_SAKURA_BUTTON_HH
#define JETSTREAM_RENDER_SAKURA_BUTTON_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct Button : public Component {
    enum class Variant {
        Default,
        Action,
        Destructive,
        Text,
    };

    struct Config {
        std::string str;
        Extent2D<F32> size = {0.0f, 0.0f};
        Variant variant = Variant::Default;
        bool disabled = false;
        std::string colorKey;
        std::string hoveredColorKey;
        std::string activeColorKey;
        std::string borderColorKey;
        std::string textColorKey;
        F32 textScale = 1.0f;
        F32 rounding = 8.0f;
        Extent2D<F32> padding = {12.0f, 5.0f};
        F32 textAlign = 0.5f;
        Callback<void()> onClick;
        Callback<void()> onDoubleClick;

        bool operator==(const Config&) const = default;
    };

    Button();
    ~Button();

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

struct NavigationItem : public Button {
    struct Config {
        std::string label;
        bool selected = false;
        Callback<void()> onSelect;
        Callback<void()> onDoubleClick;

        bool operator==(const Config&) const = default;
    };

    void update(Config config);
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_BUTTON_HH
