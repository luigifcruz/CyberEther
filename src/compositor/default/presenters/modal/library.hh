#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_LIBRARY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_LIBRARY_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/library.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct LibraryPresenter {
    const PresenterContext& context;

    explicit LibraryPresenter(const PresenterContext& context) : context(context) {}

    LibraryView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return LibraryView::Config{
            .onBrowse = [enqueue](const std::string& currentPath, std::function<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = currentPath,
                    .save = false,
                    .extensions = {"so", "so.*", "dylib", "dll"},
                    .onSelect = std::move(onSelect),
                });
            },
            .onRegister = [enqueue](const std::string& path) {
                enqueue(MailAddRegistryLibraryPath{.path = path});
            },
            .onCancel = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::Registry});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_LIBRARY_HH
