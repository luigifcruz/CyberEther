#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CALLOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CALLOUT_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct Callout : public Sakura::Div {
    enum class Tone {
        Info,
        Success,
        Warning,
        Error,
    };

    struct Config {
        std::string str;
        std::string icon;
        Tone tone = Tone::Info;
        Sakura::Text::Font font = Sakura::Text::Font::Sans;
        F32 rounding = Sakura::Rounding::Control;
    };

    void update(Config config);

 private:
    std::string toneKey() const;
    std::string toneIcon() const;

    Config config;
    Sakura::Text icon;
    Sakura::Text text;
    Sakura::Stack row;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CALLOUT_HH
