#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FLOWGRAPH_CELL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FLOWGRAPH_CELL_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphCell : public Sakura::Div {
    static constexpr F32 StarSize = 30.0f;
    static constexpr F32 CompactStarSize = 22.0f;

    struct Config {
        std::string title;
        std::string accessed;
        bool favorite = false;
        bool compact = false;
        std::string empty;
        std::function<void()> onOpen;
        std::function<void()> onToggleFavorite;
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
            .wrapped = true,
            .scale = 15.5f / Sakura::Typography::FontSize,
            .weight = empty ? 0.0f : 0.15f,
        });
        icon.update({
            .str = ICON_LC_WORKFLOW,
            .tone = Text::Tone::Secondary,
            .scale = 1.2f,
        });
        accessed.update({
            .str = this->config.accessed,
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .scale = 15.0f / Sakura::Typography::FontSize,
        });
        star.update({
            .str = ICON_LC_STAR,
            .size = {this->config.compact ? CompactStarSize : StarSize, this->config.compact ? CompactStarSize : StarSize},
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
        std::vector<Sakura::Stack::Item> lines = {{.child = &title}};
        if (!empty && !this->config.accessed.empty()) {
            lines.push_back({.child = &accessed});
        }
        texts.update({
            .spacing = 3.0f,
            .centerCross = false,
            .items = std::move(lines),
        });
        std::vector<Sakura::Stack::Item> items;
        if (!empty) {
            items.push_back({.child = &icon});
        }
        items.push_back({.child = &texts, .grow = 1.0f, .shrink = 1.0f});
        if (!empty) {
            items.push_back({.child = &star});
        }
        row.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 12.0f,
            .items = std::move(items),
        });
        Div::update({
            .content = &row,
            .padding = 12.0f,
            .rounding = 12.0f,
            .border = true,
            .chrome = true,
            .colorKey = "welcome_card_bg",
            .hoveredColorKey = empty ? "welcome_card_bg" : "welcome_card_hovered",
            .onClick = empty ? std::function<void()>() : [this]() {
                if (this->config.onOpen) {
                    this->config.onOpen();
                }
            },
        });
    }

 private:
    Config config;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Text accessed;
    Sakura::Button star;
    Sakura::Stack texts;
    Sakura::Stack row;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FLOWGRAPH_CELL_HH
