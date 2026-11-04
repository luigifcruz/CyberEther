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

    bool requested() {
        const bool asked = pending;
        pending = false;
        ready = ready || asked;
        return asked;
    }

    void update(Config config) {
        panel.update({
            .id = config.id + ":panel",
            .content = config.content,
            .size = {0.0f, 0.0f},
            .maxWidth = config.wrapWidth - 2.0f * WindowPadding,
            .transparent = true,
        });
    }

    bool prepared() const {
        return ready;
    }

    Sakura::Component* show() {
        pending = true;
        return ready ? &panel : nullptr;
    }

 private:
    static constexpr F32 WindowPadding = Sakura::Spacing::Medium;

    bool pending = false;
    bool ready = false;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_TOOLTIP_HH
