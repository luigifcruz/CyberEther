#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_PLUGIN_DETAILS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_PLUGIN_DETAILS_HH

#include "context.hh"
#include "plugin_labels.hh"

#include "../model/messages.hh"
#include "../views/plugin_details.hh"

#include "plugin_gallery.hh"
#include "tools/text.hh"

#include <algorithm>
#include <string>

namespace Jetstream {

struct PluginDetailsPresenter {
    const PresenterContext& context;

    explicit PluginDetailsPresenter(const PresenterContext& context);
    PluginDetailsView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_PLUGIN_DETAILS_HH
