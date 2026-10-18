#ifndef JETSTREAM_RENDER_SAKURA_MENU_HH
#define JETSTREAM_RENDER_SAKURA_MENU_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/root.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct MenuEntry {
    std::string label;
    std::string shortcut;
    bool selected = false;
    bool enabled = true;
    bool separator = false;
    std::string colorKey;
    Callback<void()> onClick;
    std::vector<MenuEntry> children;

    bool operator==(const MenuEntry&) const = default;
};

struct Menu : public Component {
    struct Config {
        std::vector<MenuEntry> entries;
        bool open = false;
        PopupPlacement placement = PopupPlacement::Point;
        bool atCursor = true;
        std::optional<Extent2D<F32>> point;
        std::optional<Rect> passthrough;
        Callback<void()> onClose;

        bool operator==(const Config&) const = default;
    };

    Menu();
    ~Menu();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

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

#endif  // JETSTREAM_RENDER_SAKURA_MENU_HH
