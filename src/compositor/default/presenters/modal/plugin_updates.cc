#include "plugin_updates.hh"

namespace Jetstream {

PluginUpdatesModalPresenter::PluginUpdatesModalPresenter(const PresenterContext& context) : context(context) {}

PluginPromptView::Config PluginUpdatesModalPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& plugins = context.state.plugins;

    const auto& installed = context.state.registry.plugins;
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> ids;
    for (const auto& entry : context.state.pluginUpdates()) {
        const auto info = std::find_if(installed.begin(), installed.end(), [&](const auto& item) {
            return item.id == entry.id && item.gallery && !item.temporary;
        });
        rows.push_back({
            entry.title.empty() ? entry.id : entry.title,
            info->version.empty() ? "Unknown" : info->version,
            entry.version,
        });
        ids.push_back(entry.id);
    }

    const bool single = rows.size() == 1;
    return PluginPromptView::Config{
        .title = single ? "Plugin Update Available" : "Plugin Updates Available",
        .description = jst::fmt::format("The CyberEther gallery has an update for {}. Plugins that are already "
                                        "loaded switch to the update the next time CyberEther starts. Plugins run "
                                        "native code with your permissions.",
                                        single ? "this plugin" : "these plugins"),
        .columns = {"Plugin", "Installed", "New"},
        .rows = std::move(rows),
        .cancel = "Not Now",
        .action = plugins.update.loading ? "Updating..." : (single ? "Update Plugin" : "Update Plugins"),
        .busy = plugins.update.loading,
        .onAction = [enqueue, ids = std::move(ids)]() {
            enqueue(MailUpdateGalleryPlugins{.ids = ids});
        },
        .onCancel = [enqueue]() {
            enqueue(MailCloseModal{});
        },
    };
}

}  // namespace Jetstream
