#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_ACCESS_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_ACCESS_HH

#include "component.hh"

namespace Jetstream::Sakura {

struct Access {
    static void layout(Component& root, const Context& ctx, Rect viewport) {
        root.layoutRoot(ctx, viewport);
    }

    static Result build(Component& root, Context& ctx) {
        return root.impl->buildTree(ctx);
    }

    static Result buildPending(Component& root, Context& ctx) {
        return root.impl->buildPending(ctx);
    }

    static Result paint(Component& root) {
        return root.impl->paintTree();
    }

    static bool mouseEvent(Component& root, const MouseEvent& event) {
        return root.mouseEvent(event);
    }

    static bool resourceDirty(const Component& root) {
        return root.impl->treeResourceDirty();
    }

    static bool structureDirty(const Component& root) {
        return root.impl->treeOrderDirty() || root.impl->treeStructureDirty();
    }

    static bool paintDirty(const Component& root) {
        return root.impl->isPaintDirty();
    }

    static bool layoutDirty(const Component& root) {
        return root.impl->layoutDirty || root.impl->subtreeLayoutDirty ||
               root.impl->laidEpoch != Component::Impl::layoutEpoch;
    }

    static void invalidateLayout(Component& root) {
        root.impl->invalidateLayoutTree();
    }

    static void invalidate(Component& component) {
        component.impl->invalidate(Component::Dirty::Layout);
    }

    static void invalidateSubtree(Component& root) {
        root.impl->invalidate(Component::Dirty::Layout);
        for (auto* child : root.impl->children) {
            if (child) {
                invalidateSubtree(*child);
            }
        }
    }

    static bool reachable(const Component& component) {
        return component.impl->reachable();
    }

    static bool attached(const Component& component, const Component& to) {
        return &component.impl->root() == &to.impl->root();
    }

    static bool acceptsInput(const Component& component) {
        return component.impl->root().self->acceptsInput(component);
    }

    static bool acceptsInput(const Component& host, const Component& component) {
        return host.impl->root().self->acceptsInput(component);
    }

    static const Component& root(const Component& component) {
        return *component.impl->root().self;
    }

    static std::weak_ptr<bool> alive(const Component& component) {
        return component.impl->alive;
    }

    static U64 layoutEpoch() {
        return Component::Impl::layoutEpoch;
    }

    static void advance(F64 time) {
        Component::Impl::advance(time);
    }

    static U64 layoutRuns() {
        return Component::Impl::layoutRuns;
    }

    static bool hasUnbuilt(const Component& root) {
        return root.impl->hasUnbuilt();
    }

    static void clearStructureDirty(Component& root) {
        root.impl->clearStructureDirty();
    }

    static void present(Component& root, U64 frame) {
        if (!root.impl->visible) {
            root.markOffscreen(frame);
            return;
        }
        root.wentOnscreen(frame);
        for (auto* child : root.impl->children) {
            if (child) {
                present(*child, frame);
            }
        }
    }

    static std::optional<SurfaceCursor> cursor(const Component& root, Extent2D<F32> point) {
        const auto found = root.impl->hit(point);
        if (!found.has_value()) {
            return std::nullopt;
        }
        return found->cursor;
    }

    static const Component* pointerTarget(const Component& root, Extent2D<F32> point) {
        const auto found = root.impl->hit(point);
        return found ? found->target : nullptr;
    }

    static bool key(Component& root, const KeyEvent& event) {
        return root.impl->dispatchKey(event);
    }

    static bool text(Component& root, const TextEvent& event) {
        return root.impl->dispatchText(event);
    }

    static U64 focusGeneration(const Component& root) {
        return root.impl->focusGeneration();
    }

    static void clickFocus(Component& root, std::optional<Extent2D<F32>> point, bool inside, U64 generation) {
        root.impl->clickFocus(point, inside, generation);
    }

    static bool settleFocus(Component& root) {
        return root.impl->settleFocus();
    }

    static const Component* focusOwner(const Component& root) {
        return root.impl->focusTarget();
    }

    static void releaseFocusWithin(Component& subtree) {
        const auto scope = subtree.impl->root().focusScope;
        if (scope && scope->owner && scope->owner->impl->within(&subtree)) {
            scope->owner->releaseFocus();
        }
    }

    static Component* focusedInTree(const Component& component) {
        const auto scope = component.impl->root().focusScope;
        return scope ? scope->owner : nullptr;
    }

    static bool within(const Component& component, const Component& ancestor) {
        return component.impl->within(&ancestor);
    }

    static void blur(Component& component) {
        component.releaseFocus();
    }

    static bool focus(Component& component) {
        return component.requestFocus();
    }

    static void prepareInput(Component& component) {
        component.impl->root().self->prepareInput();
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_ACCESS_HH
