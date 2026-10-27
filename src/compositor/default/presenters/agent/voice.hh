#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_VOICE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_VOICE_HH

#include "../context.hh"
#include "../../views/agent/voice.hh"

#include <algorithm>

namespace Jetstream {

inline Sakura::Voice::Phase VoicePhase(Agent::Session::Phase phase) {
    using Phase = Sakura::Voice::Phase;
    switch (phase) {
        case Agent::Session::Phase::Connecting:
            return Phase::Connecting;
        case Agent::Session::Phase::Listening:
            return Phase::Listening;
        case Agent::Session::Phase::Delegating:
            return Phase::Busy;
        case Agent::Session::Phase::Idle:
            break;
    }
    return Phase::Idle;
}

struct AgentVoicePresenter {
    const PresenterContext& context;

    AgentVoiceView::Config build() const {
        const auto& agent = context.state.agent;
        const auto enqueue = context.callbacks.enqueueMail;
        AgentVoiceView::Config config;
        config.visible = agent.ui.voiceVisible;
        if (!config.visible) {
            return config;
        }

        const auto& voice = agent.voice;
        auto& view = config.voice;
        view.phase = VoicePhase(voice.phase);
        view.level = std::clamp(std::max(voice.output * 3.0f, voice.input * 2.0f), 0.0f, 1.0f);
        view.startHint = "Click to reconnect";
        view.stopHint = "Click to end";
        view.onStop = [enqueue]() {
            enqueue(MailSetAgentVoiceVisible{.value = false});
        };
        if (view.phase == Sakura::Voice::Phase::Idle) {
            view.onStart = [enqueue]() {
                enqueue(MailStartAgentVoice{});
            };
        }
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_VOICE_HH
