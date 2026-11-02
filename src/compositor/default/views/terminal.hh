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
        Sakura::Terminal::Config terminal;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void(bool stacked)> onSplit;
    };

    void update(Config config) {
        this->config = std::move(config);
        auto view = std::move(this->config.terminal);
        const auto* screen = view.screen;
        tint = screen && screen->fill.a > 0.0f && screen->fill != screen->base ? screen->fill : ColorRGBA<F32>{};
        view.padding = Sakura::Dock::ContentPadding;
        view.rounding = Sakura::Dock::TileRounding;
        view.corners = Sakura::Box::BottomLeft | Sakura::Box::BottomRight;
        terminal.update(std::move(view));
    }

    void retire() {
        config = {};
        terminal.update({});
    }

    const std::optional<std::string>& flowgraph() const {
        return config.flowgraph;
    }

    const std::optional<std::string>& tab() const {
        return config.tab;
    }

    Sakura::Dock::Pane pane() {
        const auto split = [onSplit = config.onSplit](bool stacked) {
            return [onSplit, stacked]() {
                if (onSplit) {
                    onSplit(stacked);
                }
            };
        };
        return {
            .id = config.id,
            .title = config.title,
            .group = Group,
            .icon = ICON_LC_SQUARE_TERMINAL,
            .content = &terminal,
            .fill = tint,
            .onClose = config.onClose,
            .actions = {
                {.icon = ICON_LC_COLUMNS_2, .onClick = split(false)},
                {.icon = ICON_LC_ROWS_2, .onClick = split(true)},
            },
        };
    }

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

    void update(Config config, std::vector<Sakura::Dock::Pane> panes) {
        this->config = std::move(config);
        frame.update({
            .accepts = TerminalView::Group,
            .panes = std::move(panes),
            .tiles = this->config.tiles,
            .onTiles = this->config.onTiles,
            .loneHeader = true,
        });
    }

    Sakura::Dock::Pane pane() {
        return {
            .id = config.id,
            .title = config.title,
            .group = TerminalView::Group,
            .icon = ICON_LC_SQUARE_TERMINAL,
            .content = &frame,
            .onClose = config.onClose,
            .onFocus = config.onFocus,
        };
    }

 private:
    Config config;
    Sakura::Dock frame;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TERMINAL_HH
