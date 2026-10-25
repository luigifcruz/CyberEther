#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_GROUP_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_GROUP_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct SettingsGroup : public Sakura::Stack {
    static constexpr F32 CalloutGap = 4.0f;

    struct Config {
        std::string title;
        std::string description;
        std::string warning;
        std::vector<Sakura::SettingCard::Row> rows;
        std::string empty;
        Sakura::Component* action = nullptr;
        Sakura::Component* toolbar = nullptr;
        Sakura::Component* footer = nullptr;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        titleText.update({
            .str = config.title,
            .font = Text::Font::DisplaySemiBold,
            .scale = 1.35f,
        });
        descriptionText.update({
            .str = config.description,
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .wrapped = !config.action,
            .lineHeight = 1.05f,
        });
        warningText.update({
            .str = config.warning.empty() ? "" : ICON_LC_TRIANGLE_ALERT "  " + config.warning,
            .font = Text::Font::Sans,
            .colorKey = "banner_warning_text",
            .wrapped = true,
            .lineHeight = 1.05f,
        });
        warningPill.update({
            .content = &warningText,
            .padding = 10.0f,
            .rounding = 10.0f,
            .border = true,
            .colorKey = "banner_warning_bg",
            .hoveredColorKey = "banner_warning_bg",
            .borderColorKey = "banner_warning_border",
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
                .spacing = 3.0f,
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
            items.push_back({.child = &warningPill});
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
        items.push_back({.child = &trailing});
        Stack::update({
            .spacing = 3.0f,
            .items = std::move(items),
        });
    }

 private:
    Sakura::Text titleText;
    Sakura::Stack textColumn;
    Sakura::Spacer titleSpacing;
    Sakura::Stack titleRow;
    Sakura::Text descriptionText;
    Sakura::Text warningText;
    Sakura::Spacer warningSpacing;
    Sakura::Div warningPill;
    Sakura::Spacer toolbarSpacing;
    Sakura::Spacer cardSpacing;
    Sakura::SettingCard card;
    Sakura::Spacer footerSpacing;
    Sakura::Spacer trailing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_GROUP_HH
