#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_PLUGIN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_PLUGIN_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/plugin.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct PluginPresenter {
    const PresenterContext& context;

    explicit PluginPresenter(const PresenterContext& context) : context(context) {}

    PluginView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return PluginView::Config{
            .onBrowse = [enqueue](const std::string& currentPath, std::function<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = currentPath,
                    .save = false,
                    .extensions = {"cep"},
                    .onSelect = std::move(onSelect),
                });
            },
            .onRegister = [enqueue](const std::string& path) {
                enqueue(MailAddPluginPath{.path = path});
            },
            .onCancel = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::Registry});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_PLUGIN_HH
