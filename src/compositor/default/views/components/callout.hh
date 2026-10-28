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
    };

    void update(Config config) {
        this->config = std::move(config);
        const std::string key = toneKey();
        text.update({
            .str = jst::fmt::format("{} {}", this->config.icon.empty() ? toneIcon() : this->config.icon,
                                    this->config.str),
            .colorKey = jst::fmt::format("banner_{}_text", key),
            .wrapped = true,
        });
        Div::update({
            .content = &text,
            .padding = Sakura::Spacing::Small,
            .rounding = Sakura::Rounding::Control,
            .border = true,
            .colorKey = jst::fmt::format("banner_{}_bg", key),
            .borderColorKey = jst::fmt::format("banner_{}_border", key),
        });
    }

 private:
    std::string toneKey() const {
        switch (config.tone) {
            case Tone::Success:
                return "success";
            case Tone::Warning:
                return "warning";
            case Tone::Error:
                return "error";
            case Tone::Info:
                break;
        }
        return "info";
    }

    std::string toneIcon() const {
        switch (config.tone) {
            case Tone::Success:
                return ICON_LC_CIRCLE_CHECK;
            case Tone::Warning:
                return ICON_LC_TRIANGLE_ALERT;
            case Tone::Error:
                return ICON_LC_CIRCLE_X;
            case Tone::Info:
                break;
        }
        return ICON_LC_INFO;
    }

    Config config;
    Sakura::Text text;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CALLOUT_HH
