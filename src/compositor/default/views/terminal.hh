#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TERMINAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TERMINAL_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct TerminalView {
    static constexpr const char* Group = "terminal";

    struct Config {
        std::string id;
        std::string title;
        std::optional<std::string> flowgraph;
        std::optional<std::string> tab;
        std::optional<Sakura::Dock::Placement> placement;
        Sakura::Terminal::Config terminal;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void(bool stacked)> onSplit;
    };

    void update(Config config);
    void retire();
    const std::optional<std::string>& flowgraph() const;
    const std::optional<std::string>& tab() const;
    Sakura::Dock::Pane pane();

 private:
    Config config;
    ColorRGBA<F32> tint = {0.0f, 0.0f, 0.0f, 0.0f};
    Sakura::Terminal terminal;
};

struct TerminalTabView {
    struct Config {
        std::string id;
        std::string title;
        std::optional<Sakura::DockTiles> tiles;
        Sakura::Callback<void(std::optional<Sakura::DockTiles>)> onTiles;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void()> onFocus;
    };

    void update(Config config, std::vector<Sakura::Dock::Pane> panes);
    Sakura::Dock::Pane pane();

 private:
    Config config;
    Sakura::Dock frame;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TERMINAL_HH
