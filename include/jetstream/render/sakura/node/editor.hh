#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/render/sakura/node/types.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeEditor : public Component {
    using Child = std::function<void(const Context&)>;
    using PinRef = NodeEditorPinRef;

    struct Config {
        std::string id;
        std::string contextId;
        F32 fontScale = 0.90f;
        F32 childRounding = 12.0f;
        bool pasteEnabled = false;
        std::function<void(Extent2D<F32>, Extent2D<F32>)> onEditorContextMenu;
        std::function<void(Extent2D<F32>)> onEditorDoubleClick;
        std::function<void(PinRef, PinRef)> onLinkCreated;
        std::function<void(const std::string&)> onLinkDestroyed;
        std::function<void(const std::vector<std::string>&)> onSelectionChange;
        std::function<void(const std::vector<std::string>&)> onCopyShortcut;
        std::function<void(Extent2D<F32>)> onPasteShortcut;
        std::function<void(Extent2D<F32>)> onMouseGridPositionChange;
        std::function<void(Extent2D<F32>)> onViewportGridCenterChange;
    };

    NodeEditor();
    ~NodeEditor();

    NodeEditor(NodeEditor&&) noexcept;
    NodeEditor& operator=(NodeEditor&&) noexcept;

    NodeEditor(const NodeEditor&) = delete;
    NodeEditor& operator=(const NodeEditor&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
