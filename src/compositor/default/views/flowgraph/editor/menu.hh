#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MENU_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MENU_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphNodeMenu : public Sakura::Component {
    struct DeviceOption {
        std::string label;
        bool selected = false;
    };

    struct Config {
        std::string id;
        bool pasteEnabled = false;
        std::vector<DeviceOption> devices;
        std::function<void()> onCopy;
        std::function<void()> onPaste;
        std::function<void()> onReload;
        std::function<void()> onDelete;
        std::function<void()> onDocumentation;
        std::function<void(U64)> onDeviceSelect;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);

        popup.update({
            .id = this->config.id,
            .onClose = this->config.onClose,
        });
        copy.update({
            .id = this->config.id + ":copy",
            .label = ICON_FA_COPY " Copy Block",
            .shortcut = "CTRL+C",
            .onClick = this->config.onCopy,
        });
        paste.update({
            .id = this->config.id + ":paste",
            .label = ICON_FA_PASTE " Paste Block",
            .shortcut = "CTRL+V",
            .enabled = this->config.pasteEnabled,
            .onClick = this->config.onPaste,
        });
        deviceSeparator.update({
            .id = this->config.id + ":device-separator",
            .spacing = 0.0f,
        });
        deviceMenu.update({
            .id = this->config.id + ":device",
            .label = ICON_FA_MICROCHIP " Change Device",
            .enabled = !this->config.devices.empty(),
        });

        deviceItems.resize(this->config.devices.size());
        for (U64 i = 0; i < deviceItems.size(); ++i) {
            const auto& device = this->config.devices[i];
            deviceItems[i].update({
                .id = this->config.id + ":device:" + device.label,
                .label = device.label,
                .selected = device.selected,
                .onClick = [this, i]() {
                    if (this->config.onDeviceSelect) {
                        this->config.onDeviceSelect(i);
                    }
                },
            });
        }

        actionSeparator.update({
            .id = this->config.id + ":action-separator",
            .spacing = 0.0f,
        });
        reload.update({
            .id = this->config.id + ":reload",
            .label = ICON_FA_ARROW_ROTATE_RIGHT " Reload Block",
            .onClick = this->config.onReload,
        });
        deleteBlock.update({
            .id = this->config.id + ":delete",
            .label = ICON_FA_XMARK " Delete Block",
            .onClick = this->config.onDelete,
        });
        documentationSeparator.update({
            .id = this->config.id + ":documentation-separator",
            .spacing = 0.0f,
        });
        documentation.update({
            .id = this->config.id + ":documentation",
            .label = ICON_FA_BOOK " Documentation",
            .onClick = this->config.onDocumentation,
        });
    }

    void render(const Sakura::Context& ctx) {
        popup.render(ctx, [this](const Sakura::Context& ctx) {
            copy.render(ctx);
            paste.render(ctx);
            deviceSeparator.render(ctx);
            deviceMenu.render(ctx, [this](const Sakura::Context& ctx) {
                for (const auto& item : deviceItems) {
                    item.render(ctx);
                }
            });
            actionSeparator.render(ctx);
            reload.render(ctx);
            deleteBlock.render(ctx);
            documentationSeparator.render(ctx);
            documentation.render(ctx);
        });
    }

 private:
    Config config;
    Sakura::ContextMenu popup;
    Sakura::MenuItem copy;
    Sakura::MenuItem paste;
    Sakura::Divider deviceSeparator;
    Sakura::Menu deviceMenu;
    std::vector<Sakura::MenuItem> deviceItems;
    Sakura::Divider actionSeparator;
    Sakura::MenuItem reload;
    Sakura::MenuItem deleteBlock;
    Sakura::Divider documentationSeparator;
    Sakura::MenuItem documentation;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MENU_HH
