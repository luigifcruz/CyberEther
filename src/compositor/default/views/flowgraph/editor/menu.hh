#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MENU_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MENU_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphNodeMenu {
    struct DeviceOption {
        std::string label;
        bool selected = false;
    };

    struct Config {
        bool open = false;
        bool pasteEnabled = false;
        bool configDetachEnabled = false;
        bool configDetached = false;
        bool plotDetachEnabled = false;
        bool plotDetached = false;
        std::vector<DeviceOption> devices;
        Sakura::Callback<void()> onCopy;
        Sakura::Callback<void()> onPaste;
        Sakura::Callback<void()> onRename;
        Sakura::Callback<void()> onInspect;
        Sakura::Callback<void()> onToggleConfigDetach;
        Sakura::Callback<void()> onTogglePlotDetach;
        Sakura::Callback<void()> onReload;
        Sakura::Callback<void()> onDelete;
        Sakura::Callback<void()> onDocumentation;
        Sakura::Callback<void(U64)> onDeviceSelect;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config) {
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
                    .label = ICON_LC_COPY " Copy Block",
                    .shortcut = "CTRL+C",
                    .onClick = this->config.onCopy,
                },
                {
                    .label = ICON_LC_CLIPBOARD_PASTE " Paste Block",
                    .shortcut = "CTRL+V",
                    .enabled = this->config.pasteEnabled,
                    .onClick = this->config.onPaste,
                },
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
                    .label = ICON_LC_X " Delete Block",
                    .colorKey = "error_red",
                    .onClick = this->config.onDelete,
                },
            },
            .open = this->config.open,
            .onClose = this->config.onClose,
        });
    }

    Sakura::Component& component() {
        return menu;
    }

 private:
    Config config;
    Sakura::Menu menu;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MENU_HH
