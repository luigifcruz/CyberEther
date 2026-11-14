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
        U64 targets = 1;
        bool configDetachEnabled = false;
        bool configDetached = false;
        bool plotDetachEnabled = false;
        bool plotDetached = false;
        bool frozen = false;
        std::vector<DeviceOption> devices;
        Sakura::Callback<void()> onCopy;
        Sakura::Callback<void()> onCut;
        Sakura::Callback<void()> onPaste;
        Sakura::Callback<void()> onDuplicate;
        Sakura::Callback<void()> onRename;
        Sakura::Callback<void()> onInspect;
        Sakura::Callback<void()> onToggleConfigDetach;
        Sakura::Callback<void()> onTogglePlotDetach;
        Sakura::Callback<void()> onReload;
        Sakura::Callback<void()> onToggleFrozen;
        Sakura::Callback<void()> onDelete;
        Sakura::Callback<void()> onDocumentation;
        Sakura::Callback<void(U64)> onDeviceSelect;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    Sakura::Component& component();
    static std::string BlocksLabel(const std::string& action, U64 count);

 private:
    Config config;
    Sakura::Menu menu;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MENU_HH
