#include "missing_plugins.hh"

namespace Jetstream {

MissingPluginsModalPresenter::MissingPluginsModalPresenter(const PresenterContext& context) : context(context) {}

MissingPluginsView::Config MissingPluginsModalPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& plugins = context.state.plugins;

    return MissingPluginsView::Config{
        .plugins = plugins.missing,
        .installing = plugins.install.loading,
        .onInstall = [enqueue, ids = plugins.missing]() {
            enqueue(MailInstallGalleryPlugins{.ids = ids, .reopen = true});
        },
        .onCancel = [enqueue]() {
            enqueue(MailCloseModal{});
        },
    };
}

}  // namespace Jetstream
