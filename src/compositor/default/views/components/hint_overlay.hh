#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_HINT_OVERLAY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_HINT_OVERLAY_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct HintOverlay : public Sakura::Component {
    struct Config {
        std::string id;
        std::string icon = ICON_FA_LIGHTBULB;
        std::string title;
        std::string subtitle;
        std::vector<std::string> steps;
        std::vector<std::string> hints;
        Extent2D<F32> size = {420.0f, 340.0f};
    };

    void update(Config config) {
        this->config = std::move(config);
        overlay.update({
            .id = this->config.id + ":overlay",
            .size = this->config.size,
            .anchor = Sakura::Overlay::Anchor::Center,
        });
        card.update({
            .id = this->config.id + ":card",
            .size = this->config.size,
            .padding = 24.0f,
            .rounding = 16.0f,
            .border = true,
            .scrollbar = false,
            .mouseScroll = false,
            .inputs = false,
        });
        stack.update({
            .id = this->config.id + ":stack",
            .spacing = 8.0f,
        });
        icon.update({
            .id = this->config.id + ":icon",
            .str = this->config.icon,
            .tone = Sakura::Text::Tone::Accent,
            .align = Sakura::Text::Align::Center,
            .scale = 2.5f,
        });
        title.update({
            .id = this->config.id + ":title",
            .str = this->config.title,
            .font = Sakura::Text::Font::H2,
            .align = Sakura::Text::Align::Center,
            .scale = 1.2f,
        });
        subtitle.update({
            .id = this->config.id + ":subtitle",
            .str = this->config.subtitle,
            .tone = Sakura::Text::Tone::Secondary,
            .align = Sakura::Text::Align::Center,
        });
        stepTexts.resize(this->config.steps.size());
        for (U64 i = 0; i < this->config.steps.size(); ++i) {
            stepTexts[i].update({
                .id = this->config.id + ":step-" + std::to_string(i),
                .str = std::to_string(i + 1) + ". " + this->config.steps[i],
                .align = Sakura::Text::Align::Center,
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
                .id = this->config.id + ":hint",
                .str = "Tip: " + this->config.hints[*selectedHintIndex],
                .tone = Sakura::Text::Tone::Disabled,
                .align = Sakura::Text::Align::Center,
            });
        }
    }

    void render(const Sakura::Context& ctx) {
        overlay.render(ctx, [this](const Sakura::Context& ctx) {
            card.render(ctx, [this](const Sakura::Context& ctx) {
                Sakura::VStack::Children children = {
                    [this](const Sakura::Context& ctx) { icon.render(ctx); },
                    [this](const Sakura::Context& ctx) { title.render(ctx); },
                    [this](const Sakura::Context& ctx) { subtitle.render(ctx); },
                };
                for (U64 i = 0; i < stepTexts.size(); ++i) {
                    children.push_back([this, i](const Sakura::Context& ctx) { stepTexts[i].render(ctx); });
                }
                if (!config.hints.empty()) {
                    children.push_back([this](const Sakura::Context& ctx) { hintText.render(ctx); });
                }
                stack.render(ctx, std::move(children));
            });
        });
    }

 private:
    Config config;
    Sakura::Overlay overlay;
    Sakura::Div card;
    Sakura::VStack stack;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Text subtitle;
    std::vector<Sakura::Text> stepTexts;
    Sakura::Text hintText;
    std::optional<U64> selectedHintIndex;
    std::string selectedHintId;
    std::vector<std::string> selectedHints;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_HINT_OVERLAY_HH
