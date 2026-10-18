#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MENU_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MENU_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
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
        std::string id;
        bool open = false;
        bool pasteEnabled = false;
        bool configDetachEnabled = false;
        bool configDetached = false;
        std::vector<DeviceOption> devices;
        std::function<void()> onCopy;
        std::function<void()> onPaste;
        std::function<void()> onRename;
        std::function<void()> onInspect;
        std::function<void()> onToggleConfigDetach;
        std::function<void()> onReload;
        std::function<void()> onDelete;
        std::function<void()> onDocumentation;
        std::function<void(U64)> onDeviceSelect;
        std::function<void()> onClose;
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
                    .label = ICON_FA_COPY " Copy Block",
                    .shortcut = "CTRL+C",
                    .onClick = this->config.onCopy,
                },
                {
                    .label = ICON_FA_PASTE " Paste Block",
                    .shortcut = "CTRL+V",
                    .enabled = this->config.pasteEnabled,
                    .onClick = this->config.onPaste,
                },
                {
                    .label = ICON_FA_TAG " Rename Block",
                    .onClick = this->config.onRename,
                },
                {
                    .label = ICON_FA_CODE " Inspect & Edit",
                    .onClick = this->config.onInspect,
                },
                {
                    .label = this->config.configDetached ? ICON_FA_WINDOW_RESTORE " Attach Config"
                                                         : ICON_FA_UP_RIGHT_FROM_SQUARE " Detach Config",
                    .enabled = this->config.configDetachEnabled,
                    .onClick = this->config.onToggleConfigDetach,
                },
                {.separator = true},
                {
                    .label = ICON_FA_MICROCHIP " Change Device",
                    .enabled = !devices.empty(),
                    .children = std::move(devices),
                },
                {
                    .label = ICON_FA_ARROW_ROTATE_RIGHT " Reload Block",
                    .onClick = this->config.onReload,
                },
                {.separator = true},
                {
                    .label = ICON_FA_BOOK " Documentation",
                    .onClick = this->config.onDocumentation,
                },
                {.separator = true},
                {
                    .label = ICON_FA_XMARK " Delete Block",
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
