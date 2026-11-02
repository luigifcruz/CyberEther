#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_VOICE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_VOICE_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/titlebar.hh"

#include <algorithm>
#include <cmath>
#include <optional>

namespace Jetstream {

struct AgentVoicePresenter {
    static constexpr F32 QuietDecibels = -54.0f;
    static constexpr F32 LoudDecibels = -14.0f;

    const PresenterContext& context;

    explicit AgentVoicePresenter(const PresenterContext& context) : context(context) {}

    static F32 Loudness(F32 rms) {
        if (rms <= 0.0f) {
            return 0.0f;
        }
        const F32 decibels = 20.0f * std::log10(rms);
        return std::clamp((decibels - QuietDecibels) / (LoudDecibels - QuietDecibels), 0.0f, 1.0f);
    }

    static TitlebarView::VoiceIndicator::Phase IndicatorPhase(Agent::Session::Phase phase) {
        using Phase = TitlebarView::VoiceIndicator::Phase;
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

    std::optional<TitlebarView::VoiceIndicator> build() const {
        const auto& agent = context.state.agent;
        if (!agent.ui.voiceVisible) {
            return std::nullopt;
        }
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& voice = agent.voice;
        TitlebarView::VoiceIndicator indicator{
            .phase = IndicatorPhase(voice.phase),
            .level = std::max(Loudness(voice.input), Loudness(voice.output)),
            .onStop = [enqueue]() {
                enqueue(MailSetAgentVoiceVisible{.value = false});
            },
        };
        if (indicator.phase == TitlebarView::VoiceIndicator::Phase::Idle) {
            indicator.onStart = [enqueue]() {
                enqueue(MailStartAgentVoice{});
            };
        }
        return indicator;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_VOICE_HH
