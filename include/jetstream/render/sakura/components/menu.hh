#ifndef JETSTREAM_RENDER_SAKURA_MENU_HH
#define JETSTREAM_RENDER_SAKURA_MENU_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/popup.hh>
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

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MENU_HH
