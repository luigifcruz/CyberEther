#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CARD_BUTTON_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CARD_BUTTON_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct CardButton : public Sakura::Button {
    static constexpr F32 PillRounding = 1000.0f;

    enum class Tone {
        Card,
        Primary,
        Action,
    };

    struct Config {
        std::string str;
        Tone tone = Tone::Card;
        F32 width = Sakura::Fill;
        F32 height = 40.0f;
        F32 textScale = 1.0f;
        std::string hoverTint;
        bool tintText = false;
        Sakura::Callback<void()> onClick;
    };

    void update(Config config);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CARD_BUTTON_HH
