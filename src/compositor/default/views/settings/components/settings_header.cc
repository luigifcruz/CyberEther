#include "settings_header.hh"

namespace Jetstream {

void SettingsHeader::update(Config config) {
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

}  // namespace Jetstream
