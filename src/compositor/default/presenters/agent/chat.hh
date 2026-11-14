#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_CHAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_CHAT_HH

#include "labels.hh"
#include "transcript.hh"
#include "../context.hh"
#include "../../views/agent/chat.hh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

namespace Jetstream {

struct AgentChatPresenter {
    const PresenterContext& context;
    mutable AgentTranscriptFormatter formatter;

    explicit AgentChatPresenter(const PresenterContext& context);
    static Sakura::ChatComposer::Selector ModelSelector(const PresenterContext& context);
    AgentChatView::Config build() const;

 private:
    AgentChatPlaceholder::Config placeholder() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_CHAT_HH
