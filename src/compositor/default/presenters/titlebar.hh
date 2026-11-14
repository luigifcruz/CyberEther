#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TITLEBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TITLEBAR_HH

#include "context.hh"
#include "agent/voice.hh"

#include "../model/messages.hh"
#include "../support/terminal.hh"
#include "../views/titlebar.hh"

#include "render/sakura/titlebar.hh"

#include <memory>
#include <string>
#include <unordered_set>

namespace Jetstream {

struct TitlebarPresenter {
    const PresenterContext& context;
    AgentVoicePresenter voice;

    explicit TitlebarPresenter(const PresenterContext& context);
    TitlebarView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TITLEBAR_HH
