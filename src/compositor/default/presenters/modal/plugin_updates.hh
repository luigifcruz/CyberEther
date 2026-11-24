#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_PLUGIN_UPDATES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_PLUGIN_UPDATES_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/plugin_prompt.hh"

#include <algorithm>
#include <string>
#include <vector>

namespace Jetstream {

struct PluginUpdatesModalPresenter {
    const PresenterContext& context;

    explicit PluginUpdatesModalPresenter(const PresenterContext& context);
    PluginPromptView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_PLUGIN_UPDATES_HH
