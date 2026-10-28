#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_HEADER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_HEADER_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct SettingsHeader : public Sakura::Stack {
    struct Config {
        std::string title;
        std::string description;
        Sakura::Component* action = nullptr;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        title.update({
            .str = std::move(config.title),
            .font = Text::Font::DisplaySemiBold,
            .scale = 2.3f,
        });
        description.update({
            .str = std::move(config.description),
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .wrapped = !config.action,
            .scale = Sakura::Typography::Scale::Lead,
        });
        trailing.update({.lines = 5});
        std::vector<Item> items;
        if (config.action) {
            textColumn.update({
                .spacing = Sakura::Spacing::XSmall,
                .items = {
                    {.child = &title},
                    {.child = &description},
                },
            });
            titleRow.update({
                .axis = Axis::Horizontal,
                .items = {
                    {.child = &textColumn, .shrink = 1.0f},
                    {.child = &titleSpacing, .grow = 1.0f},
                    {.child = config.action},
                },
            });
            items = {{.child = &titleRow}};
        } else {
            items = {{.child = &title}, {.child = &description}};
        }
        items.push_back({.child = &trailing});
        Stack::update({
            .spacing = Sakura::Spacing::XSmall,
            .items = std::move(items),
        });
    }

 private:
    Sakura::Text title;
    Sakura::Stack textColumn;
    Sakura::Spacer titleSpacing;
    Sakura::Stack titleRow;
    Sakura::Text description;
    Sakura::Spacer trailing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_HEADER_HH
