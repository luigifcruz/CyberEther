#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FLOWGRAPH_CELL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FLOWGRAPH_CELL_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphCell : public Sakura::Div {
    static constexpr F32 StarSize = 22.0f;

    struct Config {
        std::string title;
        bool favorite = false;
        std::string empty;
        Sakura::Callback<void()> onOpen;
        Sakura::Callback<void()> onToggleFavorite;
        Sakura::Callback<void()> onContextMenu;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        this->config = std::move(config);
        const bool empty = !this->config.empty.empty();
        title.update({
            .str = empty ? this->config.empty : this->config.title,
            .font = Text::Font::Sans,
            .tone = empty ? Text::Tone::Secondary : Text::Tone::Primary,
            .align = empty ? Text::Align::Center : Text::Align::Left,
            .wrapped = empty,
            .ellipsis = !empty,
            .scale = Sakura::Typography::Scale::Body,
            .weight = empty ? 0.0f : 0.15f,
        });
        icon.update({
            .str = ICON_LC_WORKFLOW,
            .tone = Text::Tone::Secondary,
            .scale = Sakura::Typography::Scale::Title,
        });
        star.update({
            .str = ICON_LC_STAR,
            .size = {StarSize, StarSize},
            .colorKey = "transparent",
            .hoveredColorKey = "transparent",
            .activeColorKey = "transparent",
            .borderColorKey = "transparent",
            .textColorKey = this->config.favorite ? "favorite_star" : "text_disabled",
            .hoveredTextColorKey = this->config.favorite ? "text_secondary" : "favorite_star",
            .padding = {0.0f, 0.0f},
            .onClick = [this]() {
                if (this->config.onToggleFavorite) {
                    this->config.onToggleFavorite();
                }
            },
        });
        std::vector<Sakura::Stack::Item> items;
        if (!empty) {
            items.push_back({.child = &icon});
        }
        items.push_back({.child = &title, .grow = 1.0f, .shrink = 1.0f});
        if (!empty) {
            items.push_back({.child = &star});
        }
        row.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Medium,
            .items = std::move(items),
        });
        Div::update({
            .content = &row,
            .padding = Sakura::Spacing::Medium,
            .rounding = Sakura::Rounding::Card,
            .border = true,
            .chrome = true,
            .colorKey = "card_surface",
            .hoveredColorKey = empty ? "card_surface" : "card_surface_hovered",
            .onClick = empty ? Sakura::Callback<void()>() : [this]() {
                if (this->config.onOpen) {
                    this->config.onOpen();
                }
            },
            .onContextMenu = empty ? Sakura::Callback<void()>() : [this]() {
                if (this->config.onContextMenu) {
                    this->config.onContextMenu();
                }
            },
        });
    }

 private:
    Config config;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Button star;
    Sakura::Stack row;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FLOWGRAPH_CELL_HH
