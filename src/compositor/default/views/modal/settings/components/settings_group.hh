#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_SETTINGS_GROUP_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_SETTINGS_GROUP_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct SettingsGroup {
    struct Config {
        std::string id;
        std::string title;
        std::string description;
        std::string warning;
    };

    void update(const Config& config) {
        const auto& id = config.id;
        titleText.update({
            .id = id + "Title",
            .str = config.title,
            .font = Sakura::Text::Font::Bold,
            .scale = 1.1f,
        });
        descriptionText.update({
            .id = id + "Description",
            .str = config.description,
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });
        warningText.update({
            .id = id + "Warning",
            .str = config.warning.empty() ? "" : ICON_FA_TRIANGLE_EXCLAMATION " " + config.warning,
            .tone = Sakura::Text::Tone::Warning,
            .wrapped = true,
        });
        hasWarning = !config.warning.empty();
        card.update({
            .id = id + "Card",
            .padding = 16.0f,
            .rounding = 8.0f,
            .border = true,
            .scrollbar = false,
            .mouseScroll = false,
            .colorKey = "transparent",
            .hoveredColorKey = "transparent",
        });
        spacing.update({
            .id = id + "Spacing",
        });
    }

    void render(const Sakura::Context& ctx, const std::function<void(const Sakura::Context&)>& fields) const {
        titleText.render(ctx);
        descriptionText.render(ctx);
        if (hasWarning) {
            warningText.render(ctx);
        }
        spacing.render(ctx);
        card.render(ctx, fields);
        spacing.render(ctx);
        spacing.render(ctx);
    }

 private:
    Sakura::Text titleText;
    Sakura::Text descriptionText;
    Sakura::Text warningText;
    bool hasWarning = false;
    Sakura::Div card;
    Sakura::Spacing spacing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_SETTINGS_GROUP_HH
