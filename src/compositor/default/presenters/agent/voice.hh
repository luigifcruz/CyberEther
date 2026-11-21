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

    explicit AgentVoicePresenter(const PresenterContext& context);
    static F32 Loudness(F32 rms);
    static TitlebarView::VoiceIndicator::Phase IndicatorPhase(Agent::Session::Phase phase);
    std::optional<TitlebarView::VoiceIndicator> build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_VOICE_HH
