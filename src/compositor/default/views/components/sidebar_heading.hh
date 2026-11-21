#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_HEADING_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_HEADING_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct SidebarHeading : public Sakura::Stack {
    static constexpr F32 Inset = 9.0f;

    struct Config {
        std::string label;
        bool title = false;
    };

    void update(Config config);

 private:
    Sakura::Spacer inset;
    Sakura::Text text;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_HEADING_HH
