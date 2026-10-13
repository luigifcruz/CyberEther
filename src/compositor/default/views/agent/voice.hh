#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_VOICE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_VOICE_HH

#include "jetstream/render/sakura/components/layer.hh"
#include "jetstream/render/sakura/components/retained/canvas.hh"
#include "jetstream/render/sakura/components/retained/voice.hh"

namespace Jetstream {

struct AgentVoiceView {
    struct Config {
        bool visible = false;
        Sakura::Retained::Voice::Config voice;
    };

    AgentVoiceView() {
        canvas.mount(voice);
    }

    void update(Config config) {
        visible = config.visible;
        config.voice.id = "agent-voice";
        voice.update(std::move(config.voice));
        canvas.update({
            .id = "agent-voice:canvas",
            .size = {kSize, kSize},
            .clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
        });
        layer.update({
            .id = "agent-voice:layer",
            .anchor = Sakura::Layer::Anchor::BottomRight,
            .size = {kSize, kSize},
        });
    }

    void render(const Sakura::Context& context) {
        if (visible) {
            layer.render(context, [this](const Sakura::Context& ctx) {
                canvas.render(ctx);
            });
        }
    }

 private:
    static constexpr F32 kSize = 175.0f;

    Sakura::Retained::Voice voice;
    Sakura::Retained::Canvas canvas;
    Sakura::Layer layer;
    bool visible = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_VOICE_HH
