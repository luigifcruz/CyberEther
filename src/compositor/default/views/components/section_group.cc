#include "section_group.hh"

namespace Jetstream {

void SectionGroup::update(Config config) {
    using Text = Sakura::Text;

    titleText.update({
        .str = config.title,
        .font = Text::Font::DisplaySemiBold,
        .scale = Sakura::Typography::Scale::Heading,
    });
    descriptionText.update({
        .str = config.description,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = !config.action,
        .lineHeight = 1.05f,
    });
    warningCallout.update({
        .str = config.warning,
        .tone = Callout::Tone::Warning,
        .font = Sakura::Text::Font::Sans,
        .rounding = Sakura::Rounding::Card,
    });
    warningSpacing.update({});
    cardSpacing.update({.lines = 2});
    toolbarSpacing.update({.lines = 2});
    footerSpacing.update({.lines = 2});
    const bool hasCard = !config.rows.empty() || !config.empty.empty();
    card.update({.rows = std::move(config.rows), .empty = std::move(config.empty)});
    trailing.update({.lines = 4});

    std::vector<Item> textItems;
    if (!config.title.empty()) {
        textItems.push_back({.child = &titleText});
    }
    if (!config.description.empty()) {
        textItems.push_back({.child = &descriptionText});
    }
    std::vector<Item> items;
    if (config.action) {
        textColumn.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = std::move(textItems),
        });
        titleRow.update({
            .axis = Axis::Horizontal,
            .items = {
                {.child = &textColumn, .shrink = 1.0f},
                {.child = &titleSpacing, .grow = 1.0f},
                {.child = config.action},
            },
        });
        items.push_back({.child = &titleRow});
    } else {
        items = std::move(textItems);
    }
    if (!config.warning.empty()) {
        if (!items.empty()) {
            items.push_back({.child = &warningSpacing});
        }
        items.push_back({.child = &warningCallout});
    }
    const F32 calloutGap = config.warning.empty() ? 0.0f : CalloutGap;
    if (config.toolbar) {
        if (!items.empty()) {
            items.push_back({.child = &toolbarSpacing, .size = calloutGap});
        }
        items.push_back({.child = config.toolbar});
    }
    if (hasCard && !items.empty()) {
        items.push_back({.child = &cardSpacing, .size = config.toolbar ? 0.0f : calloutGap});
    }
    if (hasCard) {
        items.push_back({.child = &card});
    }
    if (config.footer) {
        items.push_back({.child = &footerSpacing});
        items.push_back({.child = config.footer});
    }
    if (config.trailing) {
        items.push_back({.child = &trailing});
    }
    Stack::update({
        .spacing = Sakura::Spacing::XSmall,
        .items = std::move(items),
    });
}

}  // namespace Jetstream
