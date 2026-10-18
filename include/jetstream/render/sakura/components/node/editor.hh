#ifndef JETSTREAM_RENDER_SAKURA_NODE_EDITOR_HH
#define JETSTREAM_RENDER_SAKURA_NODE_EDITOR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/node/field.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeEditor : public Component {
    enum class NodeState : U8 {
        Normal,
        Error,
        Pending,
        Loading,
    };

    enum class ResizeAxes : U8 {
        None = 0,
        X = 1 << 0,
        Y = 1 << 1,
        XY = 3,
    };

    struct PinRef {
        std::string nodeId;
        std::string pinId;
        bool isInput = false;

        bool operator==(const PinRef&) const = default;
    };

    struct Pin {
        std::string id;
        bool isInput = false;
        std::string label;
        bool enableDetach = false;
        NodeOverlay tooltip;

        bool operator==(const Pin&) const = default;
    };

    struct Geometry {
        Extent2D<F32> gridPosition;
        Extent2D<F32> contentDimensions;

        bool operator==(const Geometry&) const = default;
    };

    struct Node {
        std::string id;
        bool header = true;
        std::string title;
        std::string subtitle;
        NodeState state = NodeState::Normal;
        std::string diagnostic;
        NodeOverlay diagnosticTooltip;
        bool collapsible = false;
        bool collapsed = false;
        Callback<void()> onToggleCollapse;
        std::vector<Pin> pins;
        NodeContentStack* content = nullptr;
        ResizeAxes resize = ResizeAxes::X;
        Extent2D<F32> dimensions = {0.0f, 0.0f};
        Extent2D<F32> minimumDimensions = {120.0f, 100.0f};
        std::optional<Extent2D<F32>> gridPosition;
        std::vector<std::string> timing;
        Callback<void()> onContextMenu;
        Callback<void(const Geometry&)> onGeometryChange;

        bool operator==(const Node&) const = default;
    };

    struct Link {
        std::string id;
        PinRef start;
        PinRef end;
        bool unresolved = false;
        NodeOverlay tooltip;

        bool operator==(const Link&) const = default;
    };

    struct Config {
        std::string id;
        bool pasteEnabled = false;
        std::vector<Node> nodes;
        std::vector<Link> links;
        Callback<void(Extent2D<F32>)> onEditorContextMenu;
        Callback<void(Extent2D<F32>)> onEditorDoubleClick;
        Callback<void(PinRef, PinRef)> onLinkCreated;
        Callback<void(const std::string&)> onLinkDestroyed;
        Callback<void(const std::vector<std::string>&)> onCopyShortcut;
        Callback<void(Extent2D<F32>)> onPasteShortcut;
        Callback<void(Extent2D<F32>)> onViewportGridCenterChange;

        bool operator==(const Config&) const = default;
    };

    NodeEditor();
    ~NodeEditor();

    void update(Config config);
    F32 zoom() const;
    bool canZoom(int direction) const;
    void zoomStep(int direction);
    void zoomReset();

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool keyEvent(const KeyEvent& event) override;
    bool focusesOnClick() const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_EDITOR_HH
