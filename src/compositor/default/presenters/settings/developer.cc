#include "developer.hh"

namespace Jetstream {

DeveloperSettingsPresenter::DeveloperSettingsPresenter(const PresenterContext& context) : context(context) {}

DeveloperSettingsView::Config DeveloperSettingsPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    return DeveloperSettingsView::Config{
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

}  // namespace Jetstream
