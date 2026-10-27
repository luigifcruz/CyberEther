#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_VOICE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_VOICE_HH

#include "jetstream/render/sakura/components/hud_layer.hh"
#include "jetstream/render/sakura/components/voice.hh"

#include <utility>

namespace Jetstream {

struct AgentVoiceView {
    struct Config {
        bool visible = false;
        Sakura::Voice::Config voice;
    };

    void update(Config config) {
        visible = config.visible;
        voice.update(std::move(config.voice));
    }

    bool shown() const {
        return visible;
    }

    Sakura::HudLayer::Hud hud() {
        return {
            .content = &voice,
            .anchor = Sakura::Anchor::BottomRight,
            .margin = 0.0f,
            .padding = {0.0f, 0.0f},
            .borderSize = 0.0f,
            .rounding = 0.0f,
            .backgroundColorKey = "transparent",
            .borderColorKey = "transparent",
        };
    }

 private:
    Sakura::Voice voice;
    bool visible = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_VOICE_HH
