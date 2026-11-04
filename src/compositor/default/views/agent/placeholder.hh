#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_PLACEHOLDER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_PLACEHOLDER_HH

#include "../components/card_button.hh"

#include "jetstream/render/sakura/base.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct AgentChatPlaceholder {
    static constexpr F32 SuggestionHeight = 44.0f;
    static constexpr F32 SetupHeight = 40.0f;
    static constexpr F32 TitleScale = Sakura::Typography::Scale::Headline;
    static constexpr F32 RowSpacing = Sakura::Spacing::Small;
    static constexpr F32 SectionSpacing = Sakura::Spacing::XLarge;

    struct Suggestion {
        std::string icon;
        std::string label;
        Sakura::Callback<void()> onClick;
    };

    struct Config {
        std::string title;
        std::string subtitle;
        std::vector<Suggestion> suggestions;
        std::string setupLabel;
        Sakura::Callback<void()> onSetup;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        title.update({
            .str = config.title,
            .font = Text::Font::DisplaySemiBold,
            .wrapped = true,
            .scale = TitleScale,
            .lineHeight = 1.1f,
        });
        subtitle.update({
            .str = config.subtitle,
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .wrapped = true,
            .scale = Sakura::Typography::Scale::Caption,
            .lineHeight = 1.15f,
        });
        header.update({
            .spacing = RowSpacing,
            .centerCross = false,
            .items = {{.child = &title}, {.child = &subtitle}},
        });

        while (buttons.size() < config.suggestions.size()) {
            buttons.push_back(std::make_unique<Sakura::Button>());
        }
        std::vector<Sakura::Stack::Item> rows;
        for (U64 i = 0; i < config.suggestions.size(); ++i) {
            auto& suggestion = config.suggestions[i];
            buttons[i]->update({
                .str = jst::fmt::format("{}  {}", suggestion.icon, suggestion.label),
                .size = {Sakura::Fill, SuggestionHeight},
                .chrome = true,
                .colorKey = "agent_composer",
                .hoveredColorKey = "card_surface_hovered",
                .activeColorKey = "card_surface_hovered",
                .textColorKey = "text_secondary",
                .hoveredTextColorKey = "text_primary",
                .rounding = Sakura::Rounding::Card,
                .padding = {14.0f, 0.0f},
                .textAlign = 0.0f,
                .onClick = std::move(suggestion.onClick),
                .font = Text::Font::Sans,
            });
            rows.push_back({.child = buttons[i].get()});
        }
        if (config.onSetup) {
            setup.update({
                .str = config.setupLabel,
                .tone = CardButton::Tone::Primary,
                .height = SetupHeight,
                .textScale = 1.1f,
                .onClick = std::move(config.onSetup),
            });
            rows.push_back({.child = &setup});
        }
        actions.update({
            .spacing = RowSpacing,
            .centerCross = false,
            .items = std::move(rows),
        });
        stack.update({
            .spacing = SectionSpacing,
            .centerCross = false,
            .items = {{.child = &header}, {.child = &actions}},
        });
    }

    Sakura::Component& component() {
        return stack;
    }

 private:
    Sakura::Text title;
    Sakura::Text subtitle;
    Sakura::Stack header;
    std::vector<std::unique_ptr<Sakura::Button>> buttons;
    CardButton setup;
    Sakura::Stack actions;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_PLACEHOLDER_HH
