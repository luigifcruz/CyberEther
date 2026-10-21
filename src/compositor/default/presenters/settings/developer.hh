#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_DEVELOPER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_DEVELOPER_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/developer.hh"

namespace Jetstream {

struct DeveloperSettingsPresenter {
    const PresenterContext& context;

    explicit DeveloperSettingsPresenter(const PresenterContext& context) : context(context) {}

    DeveloperSettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return DeveloperSettingsPanel::Config{
            .timingEnabled = context.state.debug.timingEnabled,
            .logLevel = context.state.debug.logLevel,
            .onTimingEnabledChange = [enqueue](bool value) {
                enqueue(MailSetDebugTimingEnabled{.value = value});
            },
            .onLogLevelChange = [enqueue](int value) {
                enqueue(MailSetDebugLogLevel{.value = value});
            },
            .onShowBenchmarks = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Benchmark});
            },
            .onReloadPlugins = [enqueue]() {
                enqueue(MailReloadAllPlugins{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_DEVELOPER_HH
