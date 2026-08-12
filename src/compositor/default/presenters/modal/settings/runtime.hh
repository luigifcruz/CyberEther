#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_RUNTIME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_RUNTIME_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/modal/settings/runtime.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct RuntimeSettingsPresenter {
    const PresenterContext& context;

    explicit RuntimeSettingsPresenter(const PresenterContext& context) : context(context) {}

    RuntimeSettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return RuntimeSettingsPanel::Config{
            .pythonPath = context.state.runtime.pythonPath,
            .pythonCandidates = context.state.runtime.pythonCandidates,
            .pythonValidation = context.state.runtime.pythonValidation,
            .restartRequired = restartRequired(),
            .onPythonPathChange = [enqueue](const std::string& value) {
                enqueue(MailSetPythonRuntimePath{.value = value});
            },
            .onBrowsePythonPath = [enqueue](const std::string& currentPath, std::function<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = currentPath,
                    .save = false,
                    .extensions = {},
                    .onSelect = std::move(onSelect),
                });
            },
        };
    }

 private:
    bool restartRequired() const {
        const auto& current = context.state.runtime.pythonValidation;
        const auto& initial = context.state.runtime.initialPythonValidation;
        return current.valid &&
               (current.valid != initial.valid ||
                current.libraryPath != initial.libraryPath ||
                current.programPath != initial.programPath);
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_RUNTIME_HH
