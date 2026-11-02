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
        Sakura::Text::Font font = Sakura::Text::Font::Body;
    };

    void update(Config config) {
        this->config = std::move(config);
        const std::string key = toneKey();
        icon.update({
            .str = this->config.icon.empty() ? toneIcon() : this->config.icon,
            .colorKey = jst::fmt::format("banner_{}_text", key),
        });
        text.update({
            .str = this->config.str,
            .font = this->config.font,
            .colorKey = jst::fmt::format("banner_{}_text", key),
            .wrapped = true,
        });
        row.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &icon},
                {.child = &text, .grow = 1.0f, .shrink = 1.0f},
            },
        });
        Div::update({
            .content = &row,
            .padding = Sakura::Spacing::Medium,
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
    Sakura::Text icon;
    Sakura::Text text;
    Sakura::Stack row;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CALLOUT_HH
