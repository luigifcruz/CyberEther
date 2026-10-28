#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_STATUS_BADGE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_STATUS_BADGE_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct StatusBadge : public Sakura::Badge {
    enum class Tone {
        Neutral,
        Success,
        Warning,
        Error,
    };

    struct Config {
        std::string label;
        Tone tone = Tone::Neutral;
    };

    void update(Config config) {
        switch (config.tone) {
            case Tone::Neutral:
                Badge::update({
                    .label = std::move(config.label),
                    .backgroundColorKey = "header_hovered",
                    .borderColorKey = "border",
                    .textColorKey = "text_secondary",
                });
                return;
            case Tone::Success:
                return banner(std::move(config.label), "success");
            case Tone::Warning:
                return banner(std::move(config.label), "warning");
            case Tone::Error:
                return banner(std::move(config.label), "error");
        }
    }

 private:
    void banner(std::string label, const char* key) {
        Badge::update({
            .label = std::move(label),
            .backgroundColorKey = jst::fmt::format("banner_{}_bg", key),
            .borderColorKey = jst::fmt::format("banner_{}_border", key),
            .textColorKey = jst::fmt::format("banner_{}_text", key),
        });
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_STATUS_BADGE_HH
