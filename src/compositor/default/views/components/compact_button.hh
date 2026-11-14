#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_COMPACT_BUTTON_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_COMPACT_BUTTON_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct CompactButton : public Sakura::PillButton {
    enum class Tone {
        Default,
        Primary,
        Warning,
        Danger,
    };

    struct Config {
        std::string str;
        std::string icon;
        Tone tone = Tone::Default;
        bool disabled = false;
        Sakura::Callback<void()> onClick;
    };

    void update(Config config);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_COMPACT_BUTTON_HH
