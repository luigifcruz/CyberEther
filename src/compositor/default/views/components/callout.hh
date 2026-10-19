#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CALLOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CALLOUT_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"

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
        const std::string key = "banner_" + toneKey();
        text.update({
            .str = jst::fmt::format("{} {}", this->config.icon.empty() ? toneIcon() : this->config.icon,
                                    this->config.str),
            .colorKey = key + "_text",
            .wrapped = true,
        });
        Div::update({
            .content = &text,
            .padding = 8.0f,
            .rounding = 8.0f,
            .border = true,
            .colorKey = key + "_bg",
            .borderColorKey = key + "_border",
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
                return ICON_FA_CIRCLE_CHECK;
            case Tone::Warning:
                return ICON_FA_TRIANGLE_EXCLAMATION;
            case Tone::Error:
                return ICON_FA_CIRCLE_XMARK;
            case Tone::Info:
                break;
        }
        return ICON_FA_CIRCLE_INFO;
    }

    Config config;
    Sakura::Text text;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CALLOUT_HH
