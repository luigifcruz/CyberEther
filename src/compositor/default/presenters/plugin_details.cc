#include "plugin_details.hh"

namespace Jetstream {

PluginDetailsPresenter::PluginDetailsPresenter(const PresenterContext& context) : context(context) {}

PluginDetailsView::Config PluginDetailsPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& id = context.state.plugins.details;
    const auto& plugins = context.state.plugins;

    PluginDetailsView::Config config{
        .id = id,
        .title = id,
        .summary = "Loading plugin details from the CyberEther gallery.",
        .installLabel = "Install",
        .installDisabled = true,
        .onInstall = [enqueue, id]() {
            enqueue(MailInstallGalleryPlugins{.ids = {id}});
        },
        .onHowToInstall = [enqueue]() {
            enqueue(MailOpenDocs{.page = "installing-plugins"});
        },
        .onCopyLink = [enqueue, id]() {
            enqueue(MailCopyText{
                .label = "Link",
                .value = Plugin::Gallery::Host() + Plugin::Gallery::PagePath + id,
            });
        },
        .onOpenUrl = [enqueue](const std::string& url) {
            if (url.starts_with("https://") || url.starts_with("http://")) {
                enqueue(MailOpenUrl{.url = url, .notifyResult = true});
            }
        },
        .onClose = [enqueue]() {
            enqueue(MailClosePluginDetails{});
        },
    };

    const auto& entries = plugins.catalog.entries;
    const auto entry = std::find_if(entries.begin(), entries.end(), [&](const auto& item) {
        return item.id == id;
    });
    if (entry == entries.end()) {
        if (plugins.catalog.failed && !plugins.catalog.loading) {
            config.summary = "Could not reach the CyberEther gallery. Retrying shortly.";
        } else if (!plugins.catalog.loading) {
            config.summary = "This plugin is no longer in the CyberEther gallery.";
            config.installLabel = "Unavailable";
        }
        return config;
    }

    config.title = entry->title;
    config.summary = entry->summary;
    config.description = entry->description;
    const auto blocks = PluginLabels::Blocks(*entry);
    config.details = {
        {"Developer", entry->author.empty() ? "Unknown" : entry->author},
        {"Version", entry->version.empty() ? "None" : entry->version},
        {"Blocks", blocks.empty() ? "None" : Text::Join(blocks, ", ")},
        {"License", entry->license.empty() ? "Unknown" : entry->license},
        {"ID", entry->id},
    };

    if (plugins.install.loading) {
        config.installLabel = "Installing...";
    } else if (PluginLabels::Installed(context.state.registry.plugins, *entry)) {
        config.installLabel = "Installed";
    } else if (!entry->available()) {
        config.installLabel = "Unavailable";
    } else {
        config.installDisabled = false;
    }
    return config;
}

}  // namespace Jetstream
