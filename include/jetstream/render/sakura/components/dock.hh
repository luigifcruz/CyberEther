#ifndef JETSTREAM_RENDER_SAKURA_DOCK_HH
#define JETSTREAM_RENDER_SAKURA_DOCK_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/style.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct DockTiles {
    std::string pane;
    bool vertical = false;
    std::vector<F32> weights;
    std::vector<DockTiles> children;

    bool operator==(const DockTiles&) const = default;
};

struct Dock : public Component {
    static constexpr F32 ContentPadding = 6.0f;
    static constexpr F32 HeaderHeight = 20.0f;
    static constexpr F32 TileRounding = Rounding::Card;

    struct Action {
        std::string icon;
        bool active = false;
        Callback<void()> onClick;

        bool operator==(const Action&) const = default;
    };

    struct Pane {
        std::string id;
        std::string title;
        std::string group;
        std::string icon;
        Component* content = nullptr;
        bool closable = true;
        bool maximizable = true;
        bool card = true;
        bool leftmost = false;
        bool startMinimized = false;
        F32 width = 0.0f;
        F32 padding = 0.0f;
        Callback<void()> onClose;
        Callback<void()> onFocus;
        std::vector<Action> actions;

        bool operator==(const Pane&) const = default;
    };

    struct Config {
        bool tabs = false;
        bool hideTabs = false;
        std::string accepts;
        std::vector<Pane> panes;
        std::optional<DockTiles> tiles;
        std::optional<std::string> activate;
        Callback<std::string(const std::string&)> describe;
        Callback<void(std::optional<DockTiles>)> onTiles;
        Component* leading = nullptr;
        Component* trailing = nullptr;
        Component* end = nullptr;
        Component* aside = nullptr;
        F32 asideWidth = 0.0f;
        F32 tabBarHeight = 0.0f;
        F32 leadingInset = 0.0f;
        Callback<void(const std::string&, Extent2D<F32>)> onTabMenu;
        Callback<void(U8)> onBarPress;
        bool columns = false;
        std::string flexPane;
        bool loneHeader = false;
        bool headers = true;
        bool resizable = true;

        bool operator==(const Config&) const = default;
    };

    Dock();
    ~Dock();

    void update(Config config);
    bool resizing() const;
    std::string paneAt(const Extent2D<F32>& point) const;
    std::optional<Rect> tabRect(const std::string& pane) const;

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

#endif  // JETSTREAM_RENDER_SAKURA_DOCK_HH
