#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_NAVIGATION_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_NAVIGATION_HH

#include "sidebar_heading.hh"

#include "jetstream/render/sakura/base.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct SidebarNavigation : public Sakura::ScrollArea {
    static constexpr F32 Padding = 9.0f;
    static constexpr F32 ItemSpacing = 1.0f;
    static constexpr U64 SectionLines = 3;

    struct Section {
        std::string heading;
        std::vector<Sakura::Component*> items;
    };

    struct Config {
        std::string title;
        Sakura::Component* header = nullptr;
        std::vector<Section> sections;
    };

    void update(Config config) {
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

 private:
    struct Group {
        Sakura::Spacer spacing;
        SidebarHeading heading;
        Sakura::Spacer headingSpacing;
    };

    SidebarHeading title;
    Sakura::Spacer titleSpacing;
    Sakura::Spacer headerSpacing;
    std::vector<std::unique_ptr<Group>> groups;
    Sakura::Stack stack;
    Sakura::Div frame;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_NAVIGATION_HH
