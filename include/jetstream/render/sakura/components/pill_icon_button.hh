#ifndef JETSTREAM_RENDER_SAKURA_PILL_ICON_BUTTON_HH
#define JETSTREAM_RENDER_SAKURA_PILL_ICON_BUTTON_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct PillIconButton : public Component {
    struct Config {
        std::string icon;
        std::string iconColorKey = "text_primary";
        bool tinted = false;
        std::string tooltip;
        Extent2D<F32> size = {34.0f, 34.0f};
        Callback<void()> onClick;

        bool operator==(const Config&) const = default;
    };

    PillIconButton();
    ~PillIconButton();

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

#endif  // JETSTREAM_RENDER_SAKURA_PILL_ICON_BUTTON_HH
