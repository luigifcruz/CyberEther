#ifndef JETSTREAM_RENDER_SAKURA_CHROME_BUTTON_HH
#define JETSTREAM_RENDER_SAKURA_CHROME_BUTTON_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/menu.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct ChromeButton : public Component {
    enum class Glyph {
        None,
        Plus,
    };

    struct Config {
        std::string label;
        Text::Font font = Text::Font::Body;
        F32 scale = 1.0f;
        std::string colorKey;
        Glyph glyph = Glyph::None;
        F32 height = 28.0f;
        Extent2D<F32> padding = {10.0f, 0.0f};
        F32 rounding = 7.0f;
        std::string hoverColorKey = "chrome_button_hover";
        std::vector<MenuEntry> entries;
        Callback<void()> onClick;

        bool operator==(const Config&) const = default;
    };

    ChromeButton();
    ~ChromeButton();

    void update(Config config);
    bool menuOpen() const;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CHROME_BUTTON_HH
