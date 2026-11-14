#include "voice.hh"

namespace Jetstream {

AgentVoicePresenter::AgentVoicePresenter(const PresenterContext& context) : context(context) {}

F32 AgentVoicePresenter::Loudness(F32 rms) {
    if (rms <= 0.0f) {
        return 0.0f;
    }
    const F32 decibels = 20.0f * std::log10(rms);
    return std::clamp((decibels - QuietDecibels) / (LoudDecibels - QuietDecibels), 0.0f, 1.0f);
}

TitlebarView::VoiceIndicator::Phase AgentVoicePresenter::IndicatorPhase(Agent::Session::Phase phase) {
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

std::optional<TitlebarView::VoiceIndicator> AgentVoicePresenter::build() const {
    const auto& agent = context.state.agent;
    if (!agent.ui.voiceVisible) {
        return std::nullopt;
    }
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& voice = agent.voice;
    TitlebarView::VoiceIndicator indicator{
        .phase = IndicatorPhase(voice.phase),
        .level = std::max(Loudness(voice.input), Loudness(voice.output)),
        .muted = voice.muted,
        .onStop = [enqueue]() {
            enqueue(MailSetAgentVoiceVisible{.value = false});
        },
        .onMute = [enqueue, muted = voice.muted]() {
            enqueue(MailSetAgentVoiceMuted{.value = !muted});
        },
    };
    if (indicator.phase == TitlebarView::VoiceIndicator::Phase::Idle) {
        indicator.onStart = [enqueue]() {
            enqueue(MailStartAgentVoice{});
        };
    }
    return indicator;
}

}  // namespace Jetstream
