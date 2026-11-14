#include "onboarding.hh"

namespace Jetstream {

Extent2D<F32> OnboardingView::Size() {
    return {Width, 0.0f};
}

std::string OnboardingView::MajorVersion(const std::string& version) {
    const auto first = version.find('.');
    if (first == std::string::npos) {
        return version;
    }
    const auto second = version.find('.', first + 1);
    return second == std::string::npos ? version : version.substr(0, second);
}

void OnboardingView::Artwork::update(const std::string& version, const std::vector<Feature>& features) {
    using Text = Sakura::Text;

    wordmark.update({
        .str = "CyberEther",
        .font = Text::Font::Display,
        .colorKey = "cyber_blue",
        .scale = WordmarkScale,
        .emboss = WordmarkEmboss,
    });
    release.update({
        .str = "v" + MajorVersion(version),
        .font = Text::Font::Display,
        .scale = WordmarkScale,
    });
    markLeft.update({});
    markRight.update({});
    mark.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Medium,
        .items = {
            {.child = &markLeft, .grow = 1.0f},
            {.child = &wordmark},
            {.child = &release},
            {.child = &markRight, .grow = 1.0f},
        },
    });

    while (tiles.size() < features.size()) {
        tiles.push_back(std::make_unique<Tile>());
    }
    std::vector<Sakura::Stack::Item> row = {{.child = &tilesLeft, .grow = 1.0f}};
    for (U64 index = 0; index < features.size(); ++index) {
        auto& tile = *tiles[index];
        tile.icon.update({
            .str = features[index].icon,
            .colorKey = "popup_bg",
            .scale = TileIconScale,
        });
        tile.box.update({
            .content = &tile.icon,
            .size = {0.0f, 0.0f},
            .padding = Sakura::Spacing::Medium,
            .rounding = TileRounding,
            .border = false,
            .colorKey = features[index].colorKey,
            .hoveredColorKey = features[index].colorKey,
        });
        row.push_back({.child = &tile.box});
    }
    row.push_back({.child = &tilesRight, .grow = 1.0f});
    tilesLeft.update({});
    tilesRight.update({});
    tileRow.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Large,
        .items = std::move(row),
    });

    top.update({});
    bottom.update({});
    stack.update({
        .spacing = Sakura::Spacing::XLarge,
        .items = {
            {.child = &top, .grow = 1.0f},
            {.child = &mark},
            {.child = &tileRow},
            {.child = &bottom, .grow = 1.0f},
        },
    });
    Div::update({
        .content = &stack,
        .size = {Sakura::Fill, HeroHeight},
        .rounding = HeroRounding,
        .waves = true,
        .colorKey = "background",
        .hoveredColorKey = "background",
    });
}

void OnboardingView::NoteList::update(const std::vector<Section>& sections) {
    using Text = Sakura::Text;

    while (groups.size() < sections.size()) {
        groups.push_back(std::make_unique<Group>());
    }
    std::vector<Item> items;
    U64 note = 0;
    for (U64 index = 0; index < sections.size(); ++index) {
        const auto& section = sections[index];
        auto& group = *groups[index];
        std::string label = section.title;
        std::transform(label.begin(), label.end(), label.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        group.label.update({
            .str = std::move(label),
            .font = Text::Font::Display,
            .colorKey = section.colorKey,
            .scale = Sakura::Typography::Scale::Heading,
        });
        group.rule.update({.period = 30.0f, .colorKey = section.colorKey, .opacity = RuleOpacity});
        group.heading.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Medium,
            .items = {
                {.child = &group.label},
                {.child = &group.rule, .grow = 1.0f, .shrink = 1.0f},
            },
        });
        while (group.rows.size() < section.features.size()) {
            group.rows.push_back(std::make_unique<Row>());
        }
        std::vector<Item> rows;
        for (U64 entry = 0; entry < section.features.size(); ++entry) {
            const auto& feature = section.features[entry];
            auto& row = *group.rows[entry];
            row.icon.update({
                .str = feature.icon,
                .tone = Text::Tone::Secondary,
                .scale = NoteFontSize / Sakura::Typography::FontSize,
                .lineHeight = Sakura::Typography::BodyLineHeight,
            });
            row.body.update({
                .id = jst::fmt::format("OnboardingNote{}", note++),
                .value = jst::fmt::format("**{}.** {}", feature.title, feature.text),
                .fontSize = NoteFontSize,
                .padding = Sakura::Padding{0.0f, 0.0f, 0.0f, 0.0f},
                .textColorKey = "text_secondary",
                .boldColorKey = "text_primary",
            });
            row.stack.update({
                .axis = Sakura::Stack::Axis::Horizontal,
                .spacing = Sakura::Spacing::Medium,
                .centerCross = false,
                .items = {
                    {.child = &row.icon},
                    {.child = &row.body, .grow = 1.0f, .shrink = 1.0f},
                },
            });
            rows.push_back({.child = &row.stack});
        }
        group.notes.update({
            .spacing = Sakura::Spacing::Small,
            .centerCross = false,
            .items = std::move(rows),
        });
        group.stack.update({
            .spacing = Sakura::Spacing::Medium,
            .centerCross = false,
            .items = {
                {.child = &group.heading},
                {.child = &group.notes},
            },
        });
        items.push_back({.child = &group.stack});
    }
    Stack::update({
        .spacing = Sakura::Spacing::Large,
        .centerCross = false,
        .items = std::move(items),
    });
}

void OnboardingView::Page::update(const Config& config) {
    using Text = Sakura::Text;

    this->config = &config;

    icon.update({
        .str = config.icon,
        .colorKey = "text_primary",
        .scale = Sakura::Typography::Scale::Heading,
    });
    title.update({
        .str = config.title,
        .font = Text::Font::Display,
        .scale = Sakura::Typography::Scale::Headline,
    });
    titleRow.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &icon},
            {.child = &title},
        },
    });

    subtitle.update({
        .str = config.subtitle,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .scale = Sakura::Typography::Scale::Lead,
    });
    header.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &titleRow},
            {.child = &subtitle},
        },
    });

    hero.update(config.version, config.features);
    highlights.update(config.sections);

    proceed.update({
        .str = config.action,
        .tone = CardButton::Tone::Action,
        .height = ButtonHeight,
        .onClick = [this]() {
            if (this->config->onDismiss) {
                this->config->onDismiss();
            }
        },
    });

    stack.update({
        .spacing = Sakura::Spacing::Large,
        .items = {
            {.child = &header},
            {.child = &hero},
            {.child = &highlights},
            {.child = &proceed},
        },
    });
    panel.update({
        .id = "Onboarding",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& OnboardingView::Page::component() {
    return panel;
}

void OnboardingView::update(Config config) {
    this->config = std::move(config);
    page.update(this->config);
}

Sakura::Component& OnboardingView::component() {
    return page.component();
}

}  // namespace Jetstream
