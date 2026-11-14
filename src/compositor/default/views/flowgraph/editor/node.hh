#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH

#include "config/base.hh"
#include "config/stack.hh"
#include "inspect.hh"
#include "menu.hh"
#include "metrics/base.hh"

#include "../../../model/meta.hh"
#include "jetstream/block.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

using FlowgraphTooltipRows = std::vector<std::vector<std::string>>;

FlowgraphTooltipRows FlowgraphTensorLayoutRows(const Tensor& tensor);

FlowgraphTooltipRows FlowgraphTensorAttributeRows(const Tensor& tensor);

struct FlowgraphPinTooltip {
    void update(std::string id, std::string help, const std::optional<Tensor>& tensor);
    Sakura::NodeOverlay overlay();

 private:
    void prepare();

    std::string id;
    std::string help;
    bool hasTensor = false;
    FlowgraphTooltipRows layoutRows;
    FlowgraphTooltipRows attributeRows;
    U64 revision = 0;
    Tooltip tooltip;
    Sakura::Text helpText;
    Sakura::Text tensorSection;
    Sakura::Table tensorMetadataTable;
    Sakura::Text tensorAttributesSection;
    Sakura::Table tensorAttributeMetadataTable;
    Sakura::Stack content;
};

struct FlowgraphDiagnosticTooltip {
    void update(const std::string& id, const std::string& message, const std::string& colorKey);
    Sakura::NodeOverlay overlay();

 private:
    Tooltip tooltip;
    Sakura::Text headerIcon;
    Sakura::Text headerLabel;
    Sakura::Stack header;
    Sakura::Divider divider;
    Sakura::Text message;
    Sakura::Stack content;
};

struct FlowgraphNode {
    using NodeEditor = Sakura::NodeEditor;

    struct Port {
        std::string id;
        std::string label;
        std::string help;
    };

    struct Link {
        std::string block;
        std::string port;
        bool resolved = false;
        Tensor tensor;
    };

    struct Input {
        Port port;
        std::optional<Link> source;
    };

    struct Output {
        Port port;
        Tensor tensor;
    };

    struct Surface {
        std::string id;
        std::shared_ptr<const Render::Texture> texture;
        F32 height = 512.0f;
        bool detached = false;
        Sakura::Callback<void(bool)> onDetach;
        Sakura::Callback<void(const Sakura::SurfaceResize&)> onAttachedSize;
        Sakura::Callback<void(InputEvent)> onInput;
        Sakura::Callback<SurfaceCursor()> resolveCursor;
    };

    struct DeviceOption {
        std::string label;
        bool selected = false;
        DeviceType device = DeviceType::None;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider = "generic";
    };

    struct Layout {
        F32 x = 0.0f;
        F32 y = 0.0f;
        F32 width = 0.0f;
        F32 height = 0.0f;

        bool operator==(const Layout&) const = default;
    };

    struct BlockData {
        std::string name;
        std::string module;
        std::string title;
        Block::State state = Block::State::None;
        bool frozen = false;
        Block::NodeSize nodeSize = Block::NodeSize::S;
        std::string diagnostic;
        Parser::Map config;
        std::optional<Layout> layout;
        std::vector<Input> inputs;
        std::vector<Output> outputs;
        std::vector<FlowgraphMetricConfig> metrics;
        std::vector<std::string> timing;
        std::vector<FlowgraphConfigFieldConfig> configFields;
        std::vector<Surface> surfaces;
        std::vector<DeviceOption> deviceOptions;
        bool configCollapsed = false;
        bool configDetached = false;
    };

    struct Config {
        std::string id;
        std::string inspectorId;
        BlockData block;
        U64 targets = 1;
        bool timingEnabled = false;
        Sakura::Callback<void()> onCopy;
        Sakura::Callback<void()> onCut;
        Sakura::Callback<void(Extent2D<F32>)> onPaste;
        Sakura::Callback<void()> onDuplicate;
        Sakura::Callback<void()> onRename;
        Sakura::Callback<void()> onReload;
        Sakura::Callback<void()> onToggleFrozen;
        Sakura::Callback<void(Parser::Map)> onInspectApply;
        Sakura::Callback<void()> onDelete;
        Sakura::Callback<void()> onDocumentation;
        Sakura::Callback<void(DeviceType, RuntimeType, ProviderType)> onDeviceSelect;
        Sakura::Callback<void(F32, F32, F32, F32)> onLayout;
        Sakura::Callback<void(Rect, Extent2D<F32>, std::optional<Extent2D<F32>>)> onBounds;
        Sakura::Callback<void(bool)> onConfigCollapse;
        Sakura::Callback<void(bool)> onConfigDetach;
    };

    static constexpr F32 MinimumNodeWidth = 120.0f;

    static constexpr FlowgraphNodeHeightSpec SurfaceHeightSpec() {
        return {
            .policy = FlowgraphNodeHeightPolicy::FillRemaining,
            .minimum = 150.0f,
            .grow = 1.0f,
        };
    }

    void update(Config config);
    void setSurfacesDetached(bool detached);
    NodeEditor::Node takeNode();
    Sakura::Component& menuComponent();
    void appendWindows(std::vector<Sakura::FloatingLayer::Window>& windows);

 private:
    static bool Busy(const Block::State state);
    void reportBounds() const;
    void updateContent(const BlockData& block, bool isBusy, bool configHidden);

    Config config;
    Extent2D<F32> dimensions = {0.0f, 0.0f};
    std::optional<Extent2D<F32>> gridPosition;
    std::optional<NodeEditor::Geometry> geometry;
    NodeEditor::Node nodeSpec;
    Sakura::NodeContentStack content;
    FlowgraphConfigFieldStack fieldStack;
    std::vector<std::unique_ptr<FlowgraphMetricInstance>> metrics;
    std::vector<std::unique_ptr<Sakura::NodeSurface>> surfaces;
    std::vector<std::unique_ptr<FlowgraphPinTooltip>> pinTooltips;
    FlowgraphDiagnosticTooltip diagnosticTooltip;
    std::vector<FlowgraphNodeMenu::DeviceOption> deviceOptions;
    FlowgraphNodeMenu menu;
    FlowgraphNodeInspector inspector;
    bool menuOpen = false;
    bool inspectorOpen = false;
    bool layoutChanged = false;
    std::optional<Layout> emittedLayout;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
