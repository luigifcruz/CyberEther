#include "plugin_labels.hh"

namespace Jetstream {

bool PluginLabels::Installed(const std::vector<Plugin::Info>& installed, const Plugin::Gallery::Entry& entry) {
    return std::any_of(installed.begin(), installed.end(), [&](const auto& info) {
        return !info.temporary && info.gallery && info.id == entry.id &&
               (info.version == entry.version || info.pending == entry.version) &&
               info.status != Plugin::Status::Waiting && info.status != Plugin::Status::Incompatible &&
               info.status != Plugin::Status::Failed;
    });
}

bool PluginLabels::Outdated(const std::vector<Plugin::Info>& installed, const Plugin::Gallery::Entry& entry) {
    return std::any_of(installed.begin(), installed.end(), [&](const auto& info) {
        return Plugin::Gallery::Outdated(info, entry);
    });
}

std::string PluginLabels::Category(const std::vector<Plugin::Info>& installed, const Plugin::Gallery::Entry& entry) {
    if (Installed(installed, entry)) {
        return "Installed";
    }
    if (Outdated(installed, entry)) {
        return "Update Available";
    }
    if (!entry.available()) {
        return "Not Supported";
    }
    return jst::fmt::format("v{}", entry.version);
}

std::string PluginLabels::Block(const std::string& plugin, const std::string& block) {
    const auto prefix = plugin + "_";
    const auto name = block.starts_with(prefix) && block.size() > prefix.size() ? block.substr(prefix.size()) : block;
    std::string title;
    bool upper = true;
    for (const char c : name) {
        if (c == '_') {
            title.push_back(' ');
            upper = true;
            continue;
        }
        title.push_back(upper ? static_cast<char>(std::toupper(static_cast<unsigned char>(c))) : c);
        upper = false;
    }
    return title;
}

std::vector<std::string> PluginLabels::Blocks(const Plugin::Gallery::Entry& entry) {
    std::vector<std::string> titles;
    titles.reserve(entry.blocks.size());
    for (const auto& block : entry.blocks) {
        titles.push_back(Block(entry.id, block));
    }
    return titles;
}

}  // namespace Jetstream
