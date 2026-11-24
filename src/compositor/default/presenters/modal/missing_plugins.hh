#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_MISSING_PLUGINS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_MISSING_PLUGINS_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/plugin_prompt.hh"

#include "tools/text.hh"

#include <string>
#include <vector>

namespace Jetstream {

struct MissingPluginsModalPresenter {
    const PresenterContext& context;

    explicit MissingPluginsModalPresenter(const PresenterContext& context);
    PluginPromptView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_MISSING_PLUGINS_HH
