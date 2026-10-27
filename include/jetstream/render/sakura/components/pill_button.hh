#ifndef JETSTREAM_RENDER_SAKURA_PILL_BUTTON_HH
#define JETSTREAM_RENDER_SAKURA_PILL_BUTTON_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct PillButton : public Component {
    struct Config {
        std::string icon;
        std::string label;
        std::string colorKey = "text_primary";
        std::string backgroundColorKey = "pill_bg";
        std::string hoverColorKey = "pill_hover_bg";
        bool raised = false;
        bool disabled = false;
        F32 height = 28.0f;
        Callback<void()> onClick;

        bool operator==(const Config&) const = default;
    };

    PillButton();
    ~PillButton();

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

#endif  // JETSTREAM_RENDER_SAKURA_PILL_BUTTON_HH
