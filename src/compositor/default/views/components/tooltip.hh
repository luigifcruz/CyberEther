#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_TOOLTIP_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_TOOLTIP_HH

#include "jetstream/render/sakura/base.hh"

#include <string>

namespace Jetstream {

struct Tooltip {
    struct Config {
        std::string id;
        Sakura::Component* content = nullptr;
        F32 wrapWidth = 420.0f;
    };

    bool requested();
    void update(Config config);
    bool prepared() const;
    Sakura::Component* show();

 private:
    static constexpr F32 WindowPadding = Sakura::Spacing::Medium;

    bool pending = false;
    bool ready = false;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_TOOLTIP_HH
