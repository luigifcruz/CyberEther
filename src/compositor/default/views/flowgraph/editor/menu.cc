#include "menu.hh"

namespace Jetstream {

void FlowgraphNodeMenu::update(Config config) {
    using Entry = Sakura::MenuEntry;

    this->config = std::move(config);

    std::vector<Entry> devices;
    devices.reserve(this->config.devices.size());
    for (U64 i = 0; i < this->config.devices.size(); ++i) {
        const auto& device = this->config.devices[i];
        devices.push_back({
            .label = device.label,
            .selected = device.selected,
            .onClick = [this, i]() {
                if (this->config.onDeviceSelect) {
                    this->config.onDeviceSelect(i);
                }
            },
        });
    }

    menu.update({
        .entries = {
            {
                .label = BlocksLabel(ICON_LC_COPY " Copy", this->config.targets),
                .shortcut = Sakura::Shortcuts::Label(Sakura::Shortcuts::Modifier::Primary, "C"),
                .onClick = this->config.onCopy,
            },
            {
                .label = BlocksLabel(ICON_LC_SCISSORS " Cut", this->config.targets),
                .shortcut = Sakura::Shortcuts::Label(Sakura::Shortcuts::Modifier::Primary, "X"),
                .onClick = this->config.onCut,
            },
            {
                .label = ICON_LC_CLIPBOARD_PASTE " Paste",
                .shortcut = Sakura::Shortcuts::Label(Sakura::Shortcuts::Modifier::Primary, "V"),
                .onClick = this->config.onPaste,
            },
            {
                .label = BlocksLabel(ICON_LC_COPY_PLUS " Duplicate", this->config.targets),
                .shortcut = Sakura::Shortcuts::Label(Sakura::Shortcuts::Modifier::Primary, "D"),
                .onClick = this->config.onDuplicate,
            },
            {.separator = true},
            {
                .label = ICON_LC_TAG " Rename Block",
                .onClick = this->config.onRename,
            },
            {
                .label = ICON_LC_CODE " Inspect & Edit",
                .onClick = this->config.onInspect,
            },
            {
                .label = this->config.configDetached ? ICON_LC_APP_WINDOW " Attach Config"
                                                     : ICON_LC_EXTERNAL_LINK " Detach Config",
                .enabled = this->config.configDetachEnabled,
                .onClick = this->config.onToggleConfigDetach,
            },
            {
                .label = this->config.plotDetached ? ICON_LC_MINIMIZE_2 " Attach Plot"
                                                   : ICON_LC_MAXIMIZE_2 " Detach Plot",
                .enabled = this->config.plotDetachEnabled,
                .onClick = this->config.onTogglePlotDetach,
            },
            {.separator = true},
            {
                .label = ICON_LC_CPU " Change Device",
                .enabled = !devices.empty(),
                .children = std::move(devices),
            },
            {
                .label = this->config.frozen ? ICON_LC_SNOWFLAKE " Unfreeze Block"
                                             : ICON_LC_SNOWFLAKE " Freeze Block",
                .shortcut = Sakura::Shortcuts::Label(Sakura::Shortcuts::Modifier::Primary, "B"),
                .onClick = this->config.onToggleFrozen,
            },
            {
                .label = ICON_LC_ROTATE_CW " Reload Block",
                .onClick = this->config.onReload,
            },
            {.separator = true},
            {
                .label = ICON_LC_BOOK " Documentation",
                .onClick = this->config.onDocumentation,
            },
            {.separator = true},
            {
                .label = BlocksLabel(ICON_LC_X " Delete", this->config.targets),
                .shortcut = "DEL",
                .colorKey = "error_red",
                .onClick = this->config.onDelete,
            },
        },
        .open = this->config.open,
        .onClose = this->config.onClose,
    });
}

Sakura::Component& FlowgraphNodeMenu::component() {
    return menu;
}

std::string FlowgraphNodeMenu::BlocksLabel(const std::string& action, U64 count) {
    if (count > 1) {
        return jst::fmt::format("{} {} Blocks", action, count);
    }
    return action + " Block";
}

}  // namespace Jetstream
