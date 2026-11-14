#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_RUNTIME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_RUNTIME_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/runtime.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct RuntimeSettingsPresenter {
    const PresenterContext& context;

    explicit RuntimeSettingsPresenter(const PresenterContext& context);
    RuntimeSettingsView::Config build() const;

 private:
    bool restartRequired() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_RUNTIME_HH
