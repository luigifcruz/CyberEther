#include "missing_plugins.hh"

namespace Jetstream {

MissingPluginsModalPresenter::MissingPluginsModalPresenter(const PresenterContext& context) : context(context) {}

PluginPromptView::Config MissingPluginsModalPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& plugins = context.state.plugins;
    const bool single = plugins.missing.size() == 1;

    return PluginPromptView::Config{
        .title = single ? "Plugin Needed" : "Plugins Needed",
        .description = jst::fmt::format("This flowgraph uses {} {}, which {} not installed. Install {} from the "
                                        "CyberEther gallery to open it. Plugins run native code with your "
                                        "permissions.",
                                        single ? "the plugin" : "the plugins",
                                        Text::Join(plugins.missing, ", "),
                                        single ? "is" : "are",
                                        single ? "it" : "them"),
        .cancel = "Cancel",
        .action = plugins.install.loading ? "Installing..." : (single ? "Install Plugin" : "Install Plugins"),
        .busy = plugins.install.loading,
        .onAction = [enqueue, ids = plugins.missing]() {
            enqueue(MailInstallGalleryPlugins{.ids = ids, .reopen = true});
        },
        .onCancel = [enqueue]() {
            enqueue(MailCloseModal{});
        },
    };
}

}  // namespace Jetstream
