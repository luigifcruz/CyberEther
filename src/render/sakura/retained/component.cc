#include <jetstream/macros.hh>

#include "component.hh"
#include "helpers.hh"

namespace Jetstream::Sakura {

void Component::Impl::invalidate(Component::Dirty dirty) {
    paintDirty = true;
    if (dirty == Component::Dirty::Resource) {
        resourceDirty = true;
    }
    if (dirty == Component::Dirty::Tree) {
        treeDirty = true;
    }
    if (parent) {
        parent->impl->invalidate(Component::Dirty::Paint);
    }
}

void Component::Impl::setFrame(Rect newFrame) {
    if (frame == newFrame) {
        return;
    }
    frame = newFrame;
    invalidate(Component::Dirty::Paint);
}

void Component::Impl::setClip(Rect newClip) {
    clip = newClip;
}

Rect Component::Impl::childClip() const {
    return clipsChildren ? Retained::Intersect(clip, frame) : clip;
}

void Component::Impl::attachTo(Retained::Canvas* nextCanvas) {
    canvas = nextCanvas;
    for (auto* child : children) {
        if (child) {
            child->impl->attachTo(nextCanvas);
        }
    }
}

void Component::Impl::layoutChildren(const Context& ctx) {
    for (auto* child : children) {
        if (child && child->impl->visible) {
            self->layoutChild(ctx, *child, child->impl->frame);
        }
    }
}

Result Component::Impl::buildTree(Context& ctx) {
    if (visible) {
        JST_CHECK(self->build(ctx));
        resourceDirty = false;
    }
    for (auto* child : children) {
        if (!child || !child->impl->visible) {
            continue;
        }
        JST_CHECK(child->impl->buildTree(ctx));
    }
    treeDirty = false;
    return Result::SUCCESS;
}

Result Component::Impl::paintTree() {
    if (!visible) {
        return Result::SUCCESS;
    }
    JST_CHECK(self->paint());
    paintDirty = false;
    for (auto* child : children) {
        if (!child) {
            continue;
        }
        JST_CHECK(child->impl->paintTree());
    }
    return Result::SUCCESS;
}

bool Component::Impl::treeResourceDirty() const {
    if (resourceDirty || treeDirty) {
        return true;
    }
    for (auto* child : children) {
        if (child && child->impl->visible && child->impl->treeResourceDirty()) {
            return true;
        }
    }
    return false;
}

bool Component::Impl::isPaintDirty() const {
    return paintDirty;
}

Component::Component() : impl(std::make_unique<Impl>()) {
    impl->self = this;
}

Component::~Component() = default;

bool Component::visible() const {
    return impl->visible;
}

void Component::setVisible(bool visible) {
    if (impl->visible == visible) {
        return;
    }
    impl->visible = visible;
    if (impl->parent) {
        impl->parent->impl->invalidate(Dirty::Tree);
    } else {
        impl->invalidate(Dirty::Tree);
    }
}

void Component::add(Component& child) {
    child.impl->parent = this;
    child.impl->attachTo(impl->canvas);
    impl->children.push_back(&child);
    impl->invalidate(Dirty::Tree);
}

const Rect& Component::frame() const {
    return impl->frame;
}

const Rect& Component::clip() const {
    return impl->clip;
}

void Component::setClipsChildren(bool clipsChildren) {
    if (impl->clipsChildren == clipsChildren) {
        return;
    }
    impl->clipsChildren = clipsChildren;
    impl->invalidate(Dirty::Paint);
}

void Component::invalidate(Dirty dirty) {
    impl->invalidate(dirty);
}

Extent2D<F32> Component::measure(const Context&, Extent2D<F32>) {
    return {};
}

void Component::layout(const Context& ctx) {
    impl->layoutChildren(ctx);
}

Result Component::build(Context&) {
    return Result::SUCCESS;
}

Result Component::paint() {
    return Result::SUCCESS;
}

bool Component::event(const MouseEvent& event) {
    return eventChildren(event);
}

Extent2D<F32> Component::measureChild(Component& child, const Context& ctx, Extent2D<F32> available) {
    return child.measure(ctx, available);
}

void Component::layoutChild(const Context& ctx, Component& child, Rect frame) {
    child.impl->setFrame(frame);
    child.impl->setClip(impl->childClip());
    child.layout(ctx);
}

void Component::layoutChildren(const Context& ctx) {
    impl->layoutChildren(ctx);
}

bool Component::eventChildren(const MouseEvent& event) {
    for (auto it = impl->children.rbegin(); it != impl->children.rend(); ++it) {
        auto* child = *it;
        if (!child || !child->impl->visible) {
            continue;
        }
        if (event.type != MouseEventType::Move &&
            event.type != MouseEventType::Leave &&
            event.type != MouseEventType::Release &&
            !child->impl->clip.contains(event.position.x, event.position.y)) {
            continue;
        }
        if (child->event(event)) {
            return true;
        }
    }
    return false;
}

}  // namespace Jetstream::Sakura
