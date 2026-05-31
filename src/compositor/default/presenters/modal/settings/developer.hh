#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_DEVELOPER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_DEVELOPER_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/modal/settings/developer.hh"

namespace Jetstream {

struct DeveloperSettingsPresenter {
    const PresenterContext& context;

    explicit DeveloperSettingsPresenter(const PresenterContext& context) : context(context) {}

    DeveloperSettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return DeveloperSettingsPanel::Config{
            .latencyEnabled = context.state.debug.latencyEnabled,
            .timingEnabled = context.state.debug.timingEnabled,
            .logLevel = context.state.debug.logLevel,
            .onLatencyEnabledChange = [enqueue](bool value) {
                enqueue(MailSetDebugLatencyEnabled{.value = value});
            },
            .onTimingEnabledChange = [enqueue](bool value) {
                enqueue(MailSetDebugTimingEnabled{.value = value});
            },
            .onLogLevelChange = [enqueue](int value) {
                enqueue(MailSetDebugLogLevel{.value = value});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_DEVELOPER_HH
