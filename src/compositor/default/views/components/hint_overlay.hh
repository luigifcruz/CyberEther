#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_HINT_OVERLAY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_HINT_OVERLAY_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct HintOverlay {
    struct Action {
        std::string id;
        std::string label;
        Sakura::Button::Variant variant = Sakura::Button::Variant::Default;
        std::function<void()> onClick;
    };

    struct Config {
        std::string id;
        std::string icon = ICON_FA_LIGHTBULB;
        std::string title;
        std::string subtitle;
        std::vector<std::string> steps;
        std::vector<std::string> hints;
        std::vector<Action> actions;
        Extent2D<F32> size = {420.0f, 340.0f};
    };

    void update(Config config) {
        using Text = Sakura::Text;

        this->config = std::move(config);
        icon.update({
            .str = this->config.icon,
            .tone = Text::Tone::Accent,
            .align = Text::Align::Center,
            .scale = 2.5f,
        });
        title.update({
            .str = this->config.title,
            .font = Text::Font::H2,
            .align = Text::Align::Center,
            .scale = 1.2f,
        });
        subtitle.update({
            .str = this->config.subtitle,
            .tone = Text::Tone::Secondary,
            .align = Text::Align::Center,
        });
        while (stepTexts.size() < this->config.steps.size()) {
            stepTexts.push_back(std::make_unique<Text>());
        }
        for (U64 i = 0; i < this->config.steps.size(); ++i) {
            stepTexts[i]->update({
                .str = std::to_string(i + 1) + ". " + this->config.steps[i],
                .align = Text::Align::Center,
            });
        }
        if (this->config.hints.empty()) {
            selectedHintIndex.reset();
            selectedHintId.clear();
            selectedHints.clear();
        } else if (!selectedHintIndex.has_value() ||
                   selectedHintId != this->config.id ||
                   selectedHints != this->config.hints ||
                   *selectedHintIndex >= this->config.hints.size()) {
            selectedHintId = this->config.id;
            selectedHints = this->config.hints;
            selectedHintIndex = std::hash<std::string>{}(this->config.id) % this->config.hints.size();
        }
        if (!this->config.hints.empty()) {
            hintText.update({
                .str = "Tip: " + this->config.hints[*selectedHintIndex],
                .tone = Text::Tone::Disabled,
                .align = Text::Align::Center,
            });
        }
        while (actionButtons.size() < this->config.actions.size()) {
            actionButtons.push_back(std::make_unique<Sakura::Button>());
        }
        for (U64 i = 0; i < this->config.actions.size(); ++i) {
            const auto& action = this->config.actions[i];
            actionButtons[i]->update({
                .str = action.label,
                .size = {Sakura::Fill, 34.0f},
                .variant = action.variant,
                .onClick = action.onClick,
            });
        }

        std::vector<Sakura::Stack::Item> items = {
            {.child = &icon},
            {.child = &title},
            {.child = &subtitle},
        };
        for (U64 i = 0; i < this->config.steps.size(); ++i) {
            items.push_back({.child = stepTexts[i].get()});
        }
        if (!this->config.hints.empty()) {
            items.push_back({.child = &hintText});
        }
        for (U64 i = 0; i < this->config.actions.size(); ++i) {
            items.push_back({.child = actionButtons[i].get()});
        }
        stack.update({
            .spacing = 24.0f,
            .items = std::move(items),
        });
        card.update({
            .content = &stack,
            .size = this->config.size,
            .padding = 24.0f,
            .rounding = 16.0f,
            .border = true,
        });
        panel.update({
            .id = this->config.id + ":panel",
            .content = &card,
            .transparent = true,
            .backgroundColorKey = "card",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

    Extent2D<F32> size() const {
        return config.size;
    }

 private:
    Config config;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Text subtitle;
    std::vector<std::unique_ptr<Sakura::Text>> stepTexts;
    Sakura::Text hintText;
    std::vector<std::unique_ptr<Sakura::Button>> actionButtons;
    Sakura::Stack stack;
    Sakura::Div card;
    Sakura::Panel panel;
    std::optional<U64> selectedHintIndex;
    std::string selectedHintId;
    std::vector<std::string> selectedHints;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_HINT_OVERLAY_HH
