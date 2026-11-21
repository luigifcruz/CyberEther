#include "runtime.hh"

namespace Jetstream {

RuntimeSettingsPresenter::RuntimeSettingsPresenter(const PresenterContext& context) : context(context) {}

RuntimeSettingsView::Config RuntimeSettingsPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    return RuntimeSettingsView::Config{
        .pythonPath = context.state.runtime.pythonPath,
        .dependencyPolicy = context.state.runtime.dependencyPolicy,
        .pythonCandidates = context.state.runtime.pythonCandidates,
        .pythonValidation = context.state.runtime.pythonValidation,
        .restartRequired = restartRequired(),
        .onPythonPathChange = [enqueue](const std::string& value) {
            enqueue(MailSetPythonRuntimePath{.value = value});
        },
        .onDependencyPolicyChange = [enqueue](const std::string& value) {
            enqueue(MailSetRuntimeDependencyPolicy{.value = value});
        },
        .onOpenPythonEnvironmentCache = [enqueue]() {
            enqueue(MailOpenPythonEnvironmentCache{});
        },
        .onBrowsePythonPath = [enqueue](const std::string& currentPath, Sakura::Callback<void(std::string)> onSelect) {
            enqueue(MailBrowseConfigPath{
                .path = currentPath,
                .save = false,
                .extensions = {},
                .onSelect = std::move(onSelect),
            });
        },
    };
}

bool RuntimeSettingsPresenter::restartRequired() const {
    const auto& current = context.state.runtime.pythonValidation;
    const auto& initial = context.state.runtime.initialPythonValidation;
    return current.valid &&
           (current.valid != initial.valid ||
            current.libraryPath != initial.libraryPath ||
            current.programPath != initial.programPath);
}

}  // namespace Jetstream
