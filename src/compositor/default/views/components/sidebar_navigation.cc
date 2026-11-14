#include "sidebar_navigation.hh"

namespace Jetstream {

void SidebarNavigation::update(Config config) {
    title.update({.label = std::move(config.title), .title = true});
    titleSpacing.update({.lines = SectionLines});
    while (groups.size() < config.sections.size()) {
        groups.push_back(std::make_unique<Group>());
    }
    std::vector<Sakura::Stack::Item> items = {{.child = &title}, {.child = &titleSpacing}};
    if (config.header) {
        headerSpacing.update({.lines = SectionLines});
        items.push_back({.child = config.header});
        items.push_back({.child = &headerSpacing});
    }
    for (U64 index = 0; index < config.sections.size(); ++index) {
        auto& section = config.sections[index];
        auto& group = *groups[index];
        if (index > 0) {
            group.spacing.update({.lines = SectionLines});
            items.push_back({.child = &group.spacing});
        }
        group.heading.update({.label = std::move(section.heading)});
        group.headingSpacing.update({});
        items.push_back({.child = &group.heading});
        items.push_back({.child = &group.headingSpacing});
        for (auto* item : section.items) {
            items.push_back({.child = item});
        }
    }
    stack.update({
        .spacing = ItemSpacing,
        .items = std::move(items),
    });
    frame.update({
        .content = &stack,
        .padding = Padding,
        .border = false,
        .colorKey = "transparent",
    });
    ScrollArea::update({
        .content = &frame,
        .alwaysScrollbar = false,
        .border = false,
        .scrollbarInset = {0.0f, Sakura::Dock::TileRounding},
    });
}

}  // namespace Jetstream
