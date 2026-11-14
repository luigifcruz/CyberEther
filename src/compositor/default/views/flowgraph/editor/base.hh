#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_EDITOR_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_EDITOR_BASE_HH

#include "link.hh"
#include "node.hh"
#include "picker.hh"

#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphEditor {
    static constexpr F32 PillHeight = 40.0f;
    static constexpr F32 PillMargin = Sakura::Spacing::Medium;
    static constexpr F32 PillGap = Sakura::Spacing::Small;
    static constexpr F32 ZoomPillWidth = 116.0f;
    static constexpr F32 ZoomScale = Sakura::Typography::Scale::Caption;

    struct Measured {
        struct Node {
            Rect bounds;
            Extent2D<F32> size;
            std::optional<Extent2D<F32>> content;
        };

        std::unordered_map<std::string, Node> nodes;
        Rect viewport;
        std::vector<std::string> selected;
    };

    struct Config {
        std::string id;
        U64 selectionRequest = 0;
        std::vector<std::string> selection;
        bool canUndo = false;
        bool canRedo = false;
        bool debugTimingEnabled = false;
        bool measure = false;
        bool paused = false;
        Sakura::Callback<std::vector<FlowgraphBlockPicker::BlockOption>()> onBuildBlockOptions;

        std::vector<FlowgraphNode::BlockData> graph;
        std::optional<Sakura::NodeEditor::View> view;
        Sakura::Callback<void(Sakura::NodeEditor::View)> onViewChange;
        Sakura::Callback<void()> onNodesArranged;
        Sakura::Callback<void()> onUndo;
        Sakura::Callback<void()> onRedo;

        Sakura::Callback<void(const std::string&, std::optional<Extent2D<F32>>, DeviceType, RuntimeType, ProviderType)> onCreateBlock;
        Sakura::Callback<void(const std::vector<std::string>&)> onCopyBlocks;
        Sakura::Callback<void(const std::vector<std::string>&)> onCutBlocks;
        Sakura::Callback<void(std::optional<Extent2D<F32>>)> onPasteBlock;
        Sakura::Callback<void(const std::vector<std::string>&)> onDuplicateBlocks;
        Sakura::Callback<void(const std::vector<std::string>&)> onSelectBlocks;
        Sakura::Callback<void(const std::string&)> onRenameBlock;
        Sakura::Callback<void(const std::string&)> onReloadBlock;
        Sakura::Callback<void(const std::vector<std::string>&)> onToggleBlockFrozen;
        Sakura::Callback<void(const std::vector<std::string>&)> onDeleteBlocks;
        Sakura::Callback<void(const std::string&)> onOpenDocumentation;
        Sakura::Callback<void(const std::string&, DeviceType, RuntimeType, ProviderType)> onChangeBlockDevice;
        Sakura::Callback<void(const std::string&, const std::string&, const std::string&, const std::string&)> onConnectBlock;
        Sakura::Callback<void(const std::string&, const std::string&)> onDisconnectBlock;
        Sakura::Callback<void(const std::string&, Parser::Map, bool, Sakura::Callback<void(Result)>)> onReconfigureBlock;
        Sakura::Callback<void(Result, std::string)> onConfigError;
        Sakura::Callback<void(bool, std::vector<std::string>, Sakura::Callback<void(std::string)>)> onBrowseConfigPath;
        Sakura::Callback<void(const std::string&)> onOpenUrl;
        Sakura::Callback<void(const std::string&, F32, F32, F32, F32, std::optional<FlowgraphNode::Layout>)> onNodeLayout;
        Sakura::Callback<void(Measured)> onLayout;
        Sakura::Callback<void(const std::string&, bool)> onNodeConfigCollapse;
        Sakura::Callback<void(const std::string&, bool)> onNodeConfigDetach;
        Sakura::Callback<void()> onTogglePause;
        Sakura::Callback<void()> onSave;
        Sakura::Callback<void()> onToggleEnvironment;
    };

    void update(Config config);
    Sakura::Component& component();
    bool pickerOpen() const;
    void appendWindows(std::vector<Sakura::FloatingLayer::Window>& windows);

 private:
    std::vector<std::string> live(const std::vector<std::string>& ids) const;
    std::vector<std::string> targets(const std::string& blockName) const;
    static void emit(const Sakura::Callback<void(const std::vector<std::string>&)>& callback,
                     const std::vector<std::string>& blocks);
    void openBlockPicker(const Extent2D<F32>& gridPosition);
    void handleLinkCreation(const Sakura::NodeEditor::PinRef& startPin,
                            const Sakura::NodeEditor::PinRef& endPin) const;
    void handleLinkDestruction(const std::string& linkId) const;

    Config config;
    Sakura::Overlay overlay;
    Sakura::NodeEditor nodeEditor;
    FlowgraphBlockPicker blockPicker;
    Sakura::PillStepper zoomPill;
    Sakura::PillIconButton runButton;
    Sakura::PillIconButton saveButton;
    Sakura::PillIconButton addButton;
    Sakura::PillIconButton environmentButton;
    Sakura::Menu flowgraphContextMenu;
    bool flowgraphContextMenuOpen = false;
    bool arrangePending = false;
    Extent2D<F32> flowgraphContextMenuGridPos;
    Extent2D<F32> viewportGridCenter;
    Measured measured;
    std::unordered_set<std::string> liveBlocks;
    std::vector<std::string> nodeViewIds;
    std::vector<std::string> linkViewIds;
    std::unordered_map<std::string, FlowgraphLink> linkViews;
    std::unordered_map<std::string, FlowgraphNode> nodeViews;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_EDITOR_BASE_HH
