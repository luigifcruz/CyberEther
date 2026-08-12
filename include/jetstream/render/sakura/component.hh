#ifndef JETSTREAM_RENDER_SAKURA_COMPONENT_HH
#define JETSTREAM_RENDER_SAKURA_COMPONENT_HH

#include <jetstream/surface.hh>
#include <jetstream/types.hh>

#include <memory>

namespace Jetstream::Sakura {

struct Context;

namespace Retained {
struct Canvas;
}  // namespace Retained

struct Component {
    enum class Dirty {
        Paint,
        Resource,
        Tree,
    };

    Component();
    virtual ~Component();

    Component(const Component&) = delete;
    Component& operator=(const Component&) = delete;
    Component(Component&&) = delete;
    Component& operator=(Component&&) = delete;

    bool visible() const;
    void setVisible(bool visible);

 protected:
    void add(Component& child);

    const Rect& frame() const;
    const Rect& clip() const;

    void setClipsChildren(bool clipsChildren);
    void invalidate(Dirty dirty = Dirty::Paint);

    virtual Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available);
    virtual void layout(const Context& ctx);
    virtual Result build(Context& ctx);
    virtual Result paint();
    virtual bool event(const MouseEvent& event);

    Extent2D<F32> measureChild(Component& child, const Context& ctx, Extent2D<F32> available);
    void layoutChild(const Context& ctx, Component& child, Rect frame);
    void layoutChildren(const Context& ctx);
    bool eventChildren(const MouseEvent& event);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    friend struct Retained::Canvas;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_COMPONENT_HH
