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

    void update(Config config);

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
