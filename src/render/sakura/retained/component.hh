#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_COMPONENT_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_COMPONENT_HH

#include <jetstream/render/sakura/component.hh>

#include <vector>

namespace Jetstream::Sakura {

struct Component::Impl {
    Component* self = nullptr;
    Component* parent = nullptr;
    Retained::Canvas* canvas = nullptr;
    std::vector<Component*> children;

    Rect frame;
    Rect clip;
    bool visible = true;
    bool clipsChildren = false;

    bool treeDirty = true;
    bool resourceDirty = true;
    bool paintDirty = true;

    void invalidate(Component::Dirty dirty);
    void setFrame(Rect frame);
    void setClip(Rect clip);
    Rect childClip() const;
    void attachTo(Retained::Canvas* canvas);
    void layoutChildren(const Context& ctx);
    Result buildTree(Context& ctx);
    Result paintTree();
    bool treeResourceDirty() const;
    bool isPaintDirty() const;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_COMPONENT_HH
