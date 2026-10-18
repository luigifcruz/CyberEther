#ifndef JETSTREAM_RENDER_SAKURA_COMPONENT_HH
#define JETSTREAM_RENDER_SAKURA_COMPONENT_HH

#include <jetstream/render/sakura/callback.hh>
#include <jetstream/surface.hh>
#include <jetstream/types.hh>

#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace Jetstream::Sakura {

struct Context;

struct Padding {
    F32 left = 0.0f;
    F32 top = 0.0f;
    F32 right = 0.0f;
    F32 bottom = 0.0f;

    bool operator==(const Padding&) const = default;
};

constexpr F32 Fill = -1.0f;

struct Access;
struct Canvas;

struct Component {
    enum class Dirty {
        Paint,
        Resource,
        Tree,
        Order,
        Layout,
    };

    Component();
    virtual ~Component();

    Component(const Component&) = delete;
    Component& operator=(const Component&) = delete;
    Component(Component&&) = delete;
    Component& operator=(Component&&) = delete;

    bool visible() const;
    void setVisible(bool visible);
    void markOffscreen(U64 frame);

 protected:
    void addChild(Component& child);
    void removeChild(Component& child);
    void clearChildren();
    void setChildren(const std::vector<Component*>& order);
    void mountChildren(const std::vector<Component*>& active);
    void cancelMouse(Component& child);
    bool hasChild(const Component& child) const;
    const std::vector<Component*>& children() const;

    const Rect& frame() const;
    const Rect& clip() const;
    std::optional<Rect> inheritedClip() const;
    Extent2D<F32> translation() const;
    MouseEvent fromParent(const MouseEvent& event) const;

    void setClipsChildren(bool clipsChildren);
    void invalidate(Dirty dirty = Dirty::Paint);
    void requestCursor(SurfaceCursor cursor);
    void markConfigured();
    void requestFrame(F64 at = -std::numeric_limits<F64>::infinity());
    U64 layoutRevision() const;

    bool requestFocus();
    void offerFocus(bool offered = true);
    void releaseFocus();
    bool hasFocus() const;
    bool textEditorFocused() const;

    virtual Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available);
    virtual void layout(const Context& ctx);
    virtual Result build(Context& ctx);
    virtual Result paint();
    virtual void conceal();
    virtual void wentOnscreen(U64 frame);
    virtual void wentOffscreen(U64 frame);
    virtual bool mouseEvent(const MouseEvent& event);
    virtual bool keyEvent(const KeyEvent& event);
    virtual bool textEvent(const TextEvent& event);
    virtual bool shortcutEvent(const KeyEvent& event);
    virtual void focusChanged(bool focused);
    virtual bool editsText() const;
    virtual bool focusesOnClick() const;
    virtual void prepareInput();
    virtual bool acceptsInput(const Component& target) const;
    virtual std::optional<SurfaceCursor> cursorOverChildren(const Extent2D<F32>& point) const;
    virtual std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const;
    virtual bool occludes(const Extent2D<F32>& point) const;

    Extent2D<F32> measureChild(Component& child, const Context& ctx, Extent2D<F32> available);
    void layoutRoot(const Context& ctx, Rect frame);
    void layoutChild(const Context& ctx, Component& child, Rect frame);
    void layoutChild(const Context& ctx, Component& child, Rect frame, Rect clip);
    void layoutChildren(const Context& ctx);
    bool mouseEventChildren(const MouseEvent& event);
    bool deliverChild(Component& child, const MouseEvent& event);
    static bool deliver(Component& target, const MouseEvent& event);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    friend struct Access;
    friend struct Canvas;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_COMPONENT_HH
