#include <jetstream/macros.hh>

#include "component.hh"
#include "helpers.hh"
#include "popup_layer.hh"
#include "../context.hh"

#include <jetstream/render/sakura/components/canvas.hh>

#include <algorithm>

namespace Jetstream::Sakura {

namespace {

struct PendingPopup {
    Component* owner = nullptr;
    PopupRequest request;
};

Component* layingOut = nullptr;
F64 layoutTime = 0.0;
std::vector<PendingPopup> pendingPopups;

struct RequestedFrame {
    Component* component = nullptr;
    std::weak_ptr<bool> alive;
    F64 time = 0.0;
    F64 at = 0.0;
};

std::vector<RequestedFrame> requestedFrames;

}  // namespace

U64 Component::Impl::layoutRuns = 0;
U64 Component::Impl::layoutEpoch = 1;

Component* LayoutOwner() {
    return layingOut;
}

void RequestPopup(const Context& ctx, PopupRequest request) {
    if (!ctx.popups || !request.content || !layingOut) {
        return;
    }
    request.anchor.x += ctx.translation.x;
    request.anchor.y += ctx.translation.y;
    if (request.passthrough.has_value()) {
        request.passthrough->x += ctx.translation.x;
        request.passthrough->y += ctx.translation.y;
    }
    for (const auto& pending : pendingPopups) {
        if (pending.owner == layingOut && pending.request.content == request.content) {
            return;
        }
    }
    pendingPopups.push_back({.owner = layingOut, .request = std::move(request)});
}

Component::Impl::LayoutKey Component::Impl::LayoutKey::Of(const Context& ctx) {
    return {
        .pixelRatio = ctx.pixelRatio,
        .fontSize = ctx.fontSize,
        .fontRevision = ctx.render ? ctx.render->fontRevision() : 0,
        .translation = ctx.translation,
        .backdropColorKey = ctx.backdropColorKey,
        .backdropTopColorKey = ctx.backdropTopColorKey,
    };
}

bool Component::Impl::LayoutKey::matches(const Context& ctx) const {
    return pixelRatio == ctx.pixelRatio && fontSize == ctx.fontSize && fontRevision == (ctx.render ? ctx.render->fontRevision() : 0) && translation == ctx.translation && backdropColorKey == ctx.backdropColorKey &&
           backdropTopColorKey == ctx.backdropTopColorKey;
}

void Component::Impl::relayout() {
    if (layingOut != self) {
        invalidate(Component::Dirty::Layout);
    }
}

void Component::Impl::invalidate(Component::Dirty dirty) {
    if (dirty == Component::Dirty::Layout) {
        layoutDirty = true;
        ++layoutGeneration;
        markAncestors(dirty);
        return;
    }
    paintDirty = true;
    if (dirty == Component::Dirty::Resource) {
        resourceDirty = true;
    }
    if (dirty == Component::Dirty::Tree) {
        treeDirty = true;
    }
    if (dirty == Component::Dirty::Order) {
        orderDirty = true;
    }
    markAncestors(dirty);
}

void Component::Impl::markAncestors(Component::Dirty dirty) {
    if (dirty == Component::Dirty::Layout) {
        for (Component* node = parent; node; node = node->impl->parent) {
            auto& ancestor = *node->impl;
            ++ancestor.layoutGeneration;
            ancestor.subtreeLayoutDirty = true;
        }
        return;
    }
    const bool resource = dirty == Component::Dirty::Resource;
    const bool tree = dirty == Component::Dirty::Tree;
    const bool order = dirty == Component::Dirty::Order;
    const bool paintOnly = !resource && !tree && !order;
    if (paintOnly && !visible && concealed) {
        return;
    }
    for (Component* node = parent; node; node = node->impl->parent) {
        auto& ancestor = *node->impl;
        if (paintOnly && !ancestor.visible) {
            break;
        }
        const bool settled = ancestor.paintDirty &&
                             (!resource || ancestor.subtreeResourceDirty) &&
                             (!tree || ancestor.subtreeTreeDirty) &&
                             (!order || ancestor.subtreeOrderDirty);
        if (settled) {
            break;
        }
        ancestor.paintDirty = true;
        ancestor.subtreeResourceDirty |= resource;
        ancestor.subtreeTreeDirty |= tree;
        ancestor.subtreeOrderDirty |= order;
    }
}

void Component::Impl::setFrame(Rect newFrame) {
    if (frame == newFrame) {
        return;
    }
    frame = newFrame;
    invalidate(Component::Dirty::Paint);
    if (clipsChildren) {
        invalidateUnplaced();
    }
}

void Component::Impl::invalidateUnplaced() {
    for (auto* child : children) {
        if (child && !child->impl->placed) {
            child->impl->paintDirty = true;
            child->impl->invalidateUnplaced();
        }
    }
}

void Component::Impl::setClip(Rect newClip) {
    if (clip == newClip) {
        return;
    }
    clip = newClip;
    invalidate(Component::Dirty::Paint);
    invalidateUnplaced();
}

Rect Component::Impl::childClip() const {
    return clipsChildren ? Intersect(clip, frame) : clip;
}

void Component::Impl::attachTo(Canvas* nextCanvas) {
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
    JST_CHECK(self->build(ctx));
    resourceDirty = false;
    paintDirty = true;
    concealed = false;
    built = true;
    for (auto* child : children) {
        if (!child) {
            continue;
        }
        if (child->impl->deferred()) {
            child->impl->markUnbuilt();
            continue;
        }
        JST_CHECK(child->impl->buildTree(ctx));
    }
    treeDirty = false;
    orderDirty = false;
    subtreeResourceDirty = false;
    subtreeTreeDirty = false;
    subtreeOrderDirty = false;
    return Result::SUCCESS;
}

Result Component::Impl::buildPending(Context& ctx) {
    if (!built) {
        return buildTree(ctx);
    }
    for (auto* child : children) {
        if (child && !child->impl->deferred() && child->impl->hasUnbuilt()) {
            JST_CHECK(child->impl->buildPending(ctx));
            paintDirty = true;
        }
    }
    return Result::SUCCESS;
}

Result Component::Impl::paintTree() {
    if (!visible) {
        if (!concealed) {
            concealTree();
        }
        paintDirty = false;
        return Result::SUCCESS;
    }
    if (concealed) {
        invalidatePaintTree();
    }
    if (!paintDirty) {
        return Result::SUCCESS;
    }
    JST_CHECK(self->paint());
    for (auto* child : children) {
        if (!child) {
            continue;
        }
        JST_CHECK(child->impl->paintTree());
    }
    paintDirty = false;
    return Result::SUCCESS;
}

void Component::Impl::concealTree() {
    self->conceal();
    concealed = true;
    for (auto* child : children) {
        if (child) {
            child->impl->concealTree();
        }
    }
}

void Component::Impl::invalidatePaintTree() {
    paintDirty = true;
    concealed = false;
    for (auto* child : children) {
        if (child) {
            child->impl->invalidatePaintTree();
        }
    }
}

bool Component::Impl::treeResourceDirty() const {
    return resourceDirty || subtreeResourceDirty;
}

bool Component::Impl::treeOrderDirty() const {
    return orderDirty || subtreeOrderDirty;
}

bool Component::Impl::treeStructureDirty() const {
    return treeDirty || subtreeTreeDirty;
}

bool Component::Impl::unconfigured() const {
    if (!configured()) {
        return true;
    }
    for (auto* child : children) {
        if (child && child->impl->unconfigured()) {
            return true;
        }
    }
    return false;
}

bool Component::Impl::hasUnbuilt() const {
    if (!built) {
        return true;
    }
    for (auto* child : children) {
        if (child && !child->impl->deferred() && child->impl->hasUnbuilt()) {
            return true;
        }
    }
    return false;
}

bool Component::Impl::awaitingLayout() const {
    if (!visible) {
        return false;
    }
    if (!configured()) {
        return true;
    }
    for (auto* child : children) {
        if (child && child->impl->awaitingLayout()) {
            return true;
        }
    }
    return false;
}

void Component::Impl::markUnbuilt() {
    built = false;
    for (auto* child : children) {
        if (child) {
            child->impl->markUnbuilt();
        }
    }
}

void Component::Impl::clearStructureDirty() {
    treeDirty = false;
    orderDirty = false;
    subtreeTreeDirty = false;
    subtreeOrderDirty = false;
    for (auto* child : children) {
        if (child) {
            child->impl->clearStructureDirty();
        }
    }
}

bool Component::Impl::isPaintDirty() const {
    return paintDirty;
}

Component::Component() : impl(std::make_unique<Impl>()) {
    impl->self = this;
}

Component::~Component() {
    if (!impl) {
        return;
    }
    if (impl->canvas && impl->canvas->dispatching()) {
        JST_ERROR("[SAKURA] A component was destroyed while its canvas dispatched input. Queue the change and apply it outside the event.");
    }
    if (const auto handle = impl->focusHandle.lock(); handle && handle->owner == this) {
        handle->owner = nullptr;
        ++handle->generation;
    }
    if (impl->canvas && !impl->parent) {
        impl->canvas->unmount(*this);
    }
    for (auto* child : impl->children) {
        if (child && child->impl) {
            child->impl->concealTree();
            child->impl->parent = nullptr;
            child->impl->attachTo(nullptr);
        }
    }
    impl->children.clear();
    if (impl->parent && impl->parent->impl) {
        auto& siblings = impl->parent->impl->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), this), siblings.end());
        ++impl->parent->impl->childGeneration;
        impl->parent->impl->invalidate(Dirty::Tree);
        impl->parent->impl->relayout();
    }
    impl->parent = nullptr;
}

bool Component::visible() const {
    return impl->visible;
}

void Component::setVisible(bool visible) {
    if (impl->visible == visible) {
        return;
    }
    impl->visible = visible;
    impl->invalidate(visible && impl->hasUnbuilt() ? Dirty::Tree : Dirty::Paint);
    if (impl->parent) {
        impl->parent->impl->relayout();
    }
}

void Component::addChild(Component& child) {
    if (child.impl->parent == this) {
        return;
    }
    if (child.impl->parent) {
        child.impl->parent->removeChild(child);
    }
    child.impl->parent = this;
    child.impl->attachTo(impl->canvas);
    child.impl->markUnbuilt();
    impl->adoptFocus(child);
    impl->children.push_back(&child);
    ++impl->childGeneration;
    impl->invalidate(Dirty::Tree);
    impl->relayout();
}

void Component::removeChild(Component& child) {
    auto& children = impl->children;
    const auto it = std::find(children.begin(), children.end(), &child);
    if (it == children.end()) {
        return;
    }
    cancelMouse(child);
    children.erase(it);
    ++impl->childGeneration;
    child.impl->concealTree();
    child.impl->parent = nullptr;
    child.impl->attachTo(nullptr);
    impl->invalidate(Dirty::Tree);
    impl->relayout();
}

void Component::cancelMouse(Component& child) {
    MouseEvent cancel{};
    cancel.type = MouseEventType::Cancel;
    child.mouseEvent(cancel);
}

bool Component::hasChild(const Component& child) const {
    return child.impl && child.impl->parent == this;
}

const std::vector<Component*>& Component::children() const {
    return impl->children;
}

void Component::clearChildren() {
    if (impl->children.empty()) {
        return;
    }
    const std::vector<Component*> detached = impl->children;
    for (auto* child : detached) {
        if (child) {
            cancelMouse(*child);
        }
    }
    for (auto* child : impl->children) {
        if (child) {
            child->impl->concealTree();
            child->impl->parent = nullptr;
            child->impl->attachTo(nullptr);
        }
    }
    impl->children.clear();
    ++impl->childGeneration;
    impl->invalidate(Dirty::Tree);
    impl->relayout();
}

void Component::setChildren(const std::vector<Component*>& order) {
    auto& children = impl->children;
    if (order == children) {
        return;
    }
    if (order.size() == children.size() && std::is_permutation(order.begin(), order.end(), children.begin())) {
        children = order;
        ++impl->childGeneration;
        impl->invalidate(Dirty::Order);
        impl->relayout();
        return;
    }
    const std::vector<Component*> current = children;
    for (auto* child : current) {
        if (child && std::find(order.begin(), order.end(), child) == order.end()) {
            removeChild(*child);
        }
    }
    std::vector<Component*> next;
    next.reserve(order.size());
    for (auto* child : order) {
        if (std::find(next.begin(), next.end(), child) == next.end()) {
            addChild(*child);
            next.push_back(child);
        }
    }
    if (children != next) {
        children = std::move(next);
        ++impl->childGeneration;
        impl->invalidate(Dirty::Order);
        impl->relayout();
    }
}

void Component::mountChildren(const std::vector<Component*>& active) {
    const auto listed = [&active](const Component* child) {
        return std::find(active.begin(), active.end(), child) != active.end();
    };
    std::vector<Component*> next;
    std::vector<std::pair<const Component*, std::vector<Component*>>> trailing;
    const Component* anchor = nullptr;
    for (auto* child : impl->children) {
        if (listed(child)) {
            anchor = child;
            continue;
        }
        if (!anchor) {
            next.push_back(child);
            continue;
        }
        if (trailing.empty() || trailing.back().first != anchor) {
            trailing.push_back({anchor, {}});
        }
        trailing.back().second.push_back(child);
    }
    next.reserve(active.size() + impl->children.size());
    for (auto* child : active) {
        if (!child || std::find(next.begin(), next.end(), child) != next.end()) {
            continue;
        }
        next.push_back(child);
        for (const auto& [owner, hidden] : trailing) {
            if (owner == child) {
                next.insert(next.end(), hidden.begin(), hidden.end());
            }
        }
    }
    setChildren(next);
    for (auto* child : impl->children) {
        if (!listed(child) && child->impl->visible) {
            cancelMouse(*child);
            if (const auto& scope = impl->root().focusScope; scope && scope->owner) {
                for (const Component* node = scope->owner; node; node = node->impl->parent) {
                    if (node == child) {
                        scope->owner->releaseFocus();
                        break;
                    }
                }
            }
        }
        child->setVisible(listed(child));
    }
}

const Rect& Component::frame() const {
    return impl->frame;
}

Extent2D<F32> Component::Impl::translation() const {
    Extent2D<F32> total = offset;
    for (const Component* node = parent; node; node = node->impl->parent) {
        total.x += node->impl->offset.x;
        total.y += node->impl->offset.y;
    }
    return total;
}

Extent2D<F32> Component::translation() const {
    return impl->translation();
}

MouseEvent Component::fromParent(const MouseEvent& event) const {
    MouseEvent local = event;
    local.position.x -= impl->offset.x;
    local.position.y -= impl->offset.y;
    return local;
}

const Rect& Component::clip() const {
    return impl->clip;
}

std::optional<Rect> Component::inheritedClip() const {
    if (impl->placed) {
        return impl->clip;
    }
    for (const Component* node = impl->parent; node; node = node->impl->parent) {
        if (node->impl->placed) {
            return node->impl->childClip();
        }
    }
    return std::nullopt;
}

void Component::setClipsChildren(bool clipsChildren) {
    if (impl->clipsChildren == clipsChildren) {
        return;
    }
    impl->clipsChildren = clipsChildren;
    impl->invalidate(Dirty::Paint);
    impl->invalidate(Dirty::Layout);
}

void Component::invalidate(Dirty dirty) {
    impl->invalidate(dirty);
}

void Component::markConfigured() {
    impl->prepared = true;
}

void Component::requestFrame(F64 at) {
    for (auto& request : requestedFrames) {
        if (request.component == this && !request.alive.expired()) {
            request.at = std::min(request.at, at);
            return;
        }
    }
    requestedFrames.push_back({.component = this, .alive = impl->alive, .time = layoutTime, .at = at});
}

void Component::Impl::advance(F64 time) {
    std::vector<RequestedFrame> due;
    std::erase_if(requestedFrames, [&](const RequestedFrame& request) {
        if (request.alive.expired()) {
            return true;
        }
        if (request.time == time || time < request.at) {
            return false;
        }
        due.push_back(request);
        return true;
    });
    for (const auto& request : due) {
        if (!request.alive.expired()) {
            request.component->invalidate(Dirty::Layout);
        }
    }
}

U64 Component::layoutRevision() const {
    return impl->layoutGeneration;
}

void Component::requestCursor(SurfaceCursor cursor) {
    if (impl->canvas) {
        impl->canvas->requestCursor(cursor);
    }
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

void Component::conceal() {}

void Component::wentOnscreen(U64) {}

void Component::wentOffscreen(U64) {}

void Component::markOffscreen(U64 frame) {
    wentOffscreen(frame);
    for (auto* child : impl->children) {
        if (child) {
            child->markOffscreen(frame);
        }
    }
}

bool Component::mouseEvent(const MouseEvent& event) {
    return mouseEventChildren(event);
}

std::optional<SurfaceCursor> Component::cursorOverChildren(const Extent2D<F32>&) const {
    return std::nullopt;
}

std::optional<SurfaceCursor> Component::cursorAt(const Extent2D<F32>&) const {
    return std::nullopt;
}

bool Component::occludes(const Extent2D<F32>&) const {
    return false;
}

bool Component::keyEvent(const KeyEvent&) {
    return false;
}

bool Component::textEvent(const TextEvent&) {
    return false;
}

bool Component::shortcutEvent(const KeyEvent&) {
    return false;
}

void Component::focusChanged(bool) {}

bool Component::editsText() const {
    return false;
}

bool Component::focusesOnClick() const {
    return false;
}

void Component::prepareInput() {}

bool Component::acceptsInput(const Component&) const {
    return true;
}

bool Component::requestFocus() {
    const std::weak_ptr<bool> selfToken = impl->alive;
    auto* inputRoot = impl->root().self;
    const std::weak_ptr<bool> rootToken = inputRoot->impl->alive;
    inputRoot->prepareInput();
    if (selfToken.expired() || rootToken.expired() || impl->root().self != inputRoot) {
        return false;
    }
    if (!impl->root().self->acceptsInput(*this)) {
        return false;
    }
    const auto scope = impl->scope();
    if (scope->owner == this) {
        return true;
    }
    if (const auto previous = impl->focusHandle.lock(); previous && previous != scope && previous->owner == this) {
        previous->owner = nullptr;
        ++previous->generation;
    }
    const std::weak_ptr<bool> token = impl->alive;
    if (Component* previous = scope->owner) {
        scope->owner = nullptr;
        const U64 generation = ++scope->generation;
        previous->impl->focusHandle.reset();
        previous->focusChanged(false);
        if (token.expired() || rootToken.expired()) {
            return false;
        }
        if (impl->root().self != inputRoot || impl->scope() != scope || !inputRoot->acceptsInput(*this)) {
            return false;
        }
        if (scope->owner || scope->generation != generation) {
            return true;
        }
    }
    scope->owner = this;
    ++scope->generation;
    impl->focusHandle = scope;
    focusChanged(true);
    return !token.expired() && !rootToken.expired();
}

void Component::offerFocus(bool offered) {
    if (offered && !impl->focusOffered) {
        impl->scope()->offers.push_back({this, impl->alive});
    }
    impl->focusOffered = offered;
}

void Component::releaseFocus() {
    const auto scope = impl->focusHandle.lock();
    if (!scope || scope->owner != this) {
        return;
    }
    scope->owner = nullptr;
    ++scope->generation;
    impl->focusHandle.reset();
    focusChanged(false);
}

bool Component::hasFocus() const {
    const auto scope = impl->focusHandle.lock();
    return scope && scope->owner == this;
}

bool Component::textEditorFocused() const {
    const auto& scope = impl->root().focusScope;
    return scope && scope->owner && scope->owner->editsText();
}

std::optional<Component::Impl::Hit> Component::Impl::hit(Extent2D<F32> point) const {
    if (!visible || !clip.contains(point.x, point.y)) {
        return std::nullopt;
    }
    const bool allowed = root().self->acceptsInput(*self);
    if (allowed) {
        if (const auto cursor = self->cursorOverChildren(point)) {
            return Hit{self, *cursor};
        }
    }
    for (auto it = children.rbegin(); it != children.rend(); ++it) {
        const Component* child = *it;
        if (!child) {
            continue;
        }
        const Extent2D<F32> local = {point.x - child->impl->offset.x, point.y - child->impl->offset.y};
        if (const auto found = child->impl->hit(local)) {
            return found;
        }
        if (child->impl->visible && root().self->acceptsInput(*child) && child->occludes(local)) {
            return Hit{child, SurfaceCursor::Default};
        }
    }
    if (allowed) {
        if (const auto cursor = self->cursorAt(point)) {
            return Hit{self, *cursor};
        }
    }
    return std::nullopt;
}

bool Component::deliver(Component& target, const MouseEvent& event) {
    return target.mouseEvent(target.fromParent(event));
}

Component::Impl& Component::Impl::root() {
    Impl* node = this;
    while (node->parent) {
        node = node->parent->impl.get();
    }
    return *node;
}

const Component::Impl& Component::Impl::root() const {
    const Impl* node = this;
    while (node->parent) {
        node = node->parent->impl.get();
    }
    return *node;
}

std::shared_ptr<Component::Impl::FocusScope> Component::Impl::scope() {
    auto& top = root();
    if (!top.focusScope) {
        top.focusScope = std::make_shared<FocusScope>();
    }
    return top.focusScope;
}

bool Component::Impl::within(const Component* ancestor) const {
    for (const Component* node = self; node; node = node->impl->parent) {
        if (node == ancestor) {
            return true;
        }
    }
    return false;
}

bool Component::Impl::reachable() const {
    for (const Component* node = self; node; node = node->impl->parent) {
        if (!node->impl->visible) {
            return false;
        }
    }
    return true;
}

void Component::Impl::adoptFocus(Component& child) {
    const auto adopted = std::move(child.impl->focusScope);
    if (!adopted) {
        return;
    }
    const auto target = scope();
    target->offers.insert(target->offers.end(), adopted->offers.begin(), adopted->offers.end());
    if (!adopted->owner) {
        return;
    }
    Component* owner = adopted->owner;
    if (target->owner) {
        adopted->owner = nullptr;
        owner->impl->focusHandle.reset();
        owner->focusChanged(false);
        return;
    }
    target->owner = owner;
    ++target->generation;
    owner->impl->focusHandle = target;
}

Component* Component::Impl::focusTarget() const {
    const auto& scope = focusScope;
    if (!scope || !scope->owner) {
        return nullptr;
    }
    Component* owner = scope->owner;
    if (!owner->impl->within(self) || !owner->impl->reachable() || !root().self->acceptsInput(*owner)) {
        return nullptr;
    }
    return owner;
}

namespace {

using Chain = std::vector<std::pair<Component*, std::weak_ptr<bool>>>;

}  // namespace

template<typename Handler>
bool Component::Impl::bubble(Handler&& handler) {
    const auto* inputRoot = root().self;
    const std::weak_ptr<bool> rootToken = inputRoot->impl->alive;
    Component* target = focusTarget();
    Chain chain;
    for (Component* node = target; node; node = node->impl->parent) {
        chain.push_back({node, node->impl->alive});
        if (node == self) {
            break;
        }
    }
    for (const auto& [node, token] : chain) {
        if (token.expired() || rootToken.expired()) {
            return true;
        }
        if ((inputRoot->acceptsInput(*node) && handler(*node)) || token.expired()) {
            return true;
        }
    }
    return false;
}

bool Component::Impl::dispatchKey(const KeyEvent& event) {
    const std::weak_ptr<bool> token = alive;
    auto* inputRoot = root().self;
    const std::weak_ptr<bool> rootToken = inputRoot->impl->alive;
    inputRoot->prepareInput();
    if (token.expired() || rootToken.expired()) {
        return true;
    }
    if (bubble([&](Component& node) { return node.keyEvent(event); })) {
        return true;
    }
    return !token.expired() && dispatchShortcut(event);
}

bool Component::Impl::dispatchText(const TextEvent& event) {
    const std::weak_ptr<bool> token = alive;
    auto* inputRoot = root().self;
    const std::weak_ptr<bool> rootToken = inputRoot->impl->alive;
    inputRoot->prepareInput();
    if (token.expired() || rootToken.expired()) {
        return true;
    }
    return bubble([&](Component& node) { return node.textEvent(event); });
}

bool Component::Impl::dispatchShortcut(const KeyEvent& event) {
    if (!visible) {
        return false;
    }
    Chain snapshot;
    snapshot.reserve(children.size());
    for (auto* child : children) {
        if (child) {
            snapshot.push_back({child, child->impl->alive});
        }
    }
    const std::weak_ptr<bool> token = alive;
    for (auto it = snapshot.rbegin(); it != snapshot.rend(); ++it) {
        if (it->second.expired()) {
            continue;
        }
        if (it->first->impl->dispatchShortcut(event) || it->second.expired() || token.expired()) {
            return true;
        }
    }
    const bool handled = root().self->acceptsInput(*self) && self->shortcutEvent(event);
    return handled || token.expired();
}

U64 Component::Impl::focusGeneration() const {
    return focusScope ? focusScope->generation : 0;
}

void Component::Impl::clickFocus(std::optional<Extent2D<F32>> point, bool inside, U64 generation) {
    if (focusGeneration() != generation) {
        return;
    }
    Component* owner = focusScope ? focusScope->owner : nullptr;
    const Component* target = nullptr;
    if (inside && point.has_value()) {
        if (const auto found = hit(*point)) {
            target = found->target;
        }
    }
    if (owner && target && target->impl->within(owner)) {
        return;
    }
    Component* claimant = nullptr;
    for (const Component* node = target; node && !claimant; node = node->impl->parent) {
        if (node->focusesOnClick()) {
            claimant = const_cast<Component*>(node);
        }
    }
    if (!target && inside && point.has_value()) {
        claimant = claimantAt(*point);
    }
    if (claimant) {
        claimant->requestFocus();
    } else if (owner) {
        owner->releaseFocus();
    }
}

Component* Component::Impl::claimantAt(Extent2D<F32> point) const {
    if (!visible || !clip.contains(point.x, point.y)) {
        return nullptr;
    }
    for (auto it = children.rbegin(); it != children.rend(); ++it) {
        const Component* child = *it;
        if (!child) {
            continue;
        }
        const Extent2D<F32> local = {point.x - child->impl->offset.x, point.y - child->impl->offset.y};
        if (Component* found = child->impl->claimantAt(local)) {
            return found;
        }
    }
    return self->focusesOnClick() && frame.contains(point.x, point.y) ? self : nullptr;
}

bool Component::Impl::settleFocus() {
    if (!focusScope || focusScope->offers.empty()) {
        return false;
    }
    const auto scope = focusScope;
    const std::weak_ptr<bool> rootToken = alive;
    auto pending = std::move(scope->offers);
    scope->offers.clear();
    bool changed = false;
    for (const auto& [component, token] : pending) {
        if (rootToken.expired()) {
            return true;
        }
        if (token.expired() || !component->impl->focusOffered) {
            continue;
        }
        auto& state = *component->impl;
        if (!state.within(self) || !state.reachable() || !state.placed || state.frame.empty()) {
            scope->offers.push_back({component, token});
            continue;
        }
        state.focusOffered = false;
        if (!component->hasFocus() && !component->textEditorFocused()) {
            component->requestFocus();
            changed = true;
        }
    }
    return changed;
}

Extent2D<F32> Component::measureChild(Component& child, const Context& ctx, Extent2D<F32> available) {
    return child.impl->measureCached(ctx, available);
}

Extent2D<F32> Component::Impl::measureCached(const Context& ctx, Extent2D<F32> available) {
    for (const auto& memo : measureMemos) {
        if (memo.epoch == layoutEpoch && memo.generation == layoutGeneration && memo.available == available &&
            memo.frame == frame &&
            memo.key.matches(ctx)) {
            return memo.result;
        }
    }
    const U64 generation = layoutGeneration;
    const Extent2D<F32> result = self->measure(ctx, available);
    MeasureMemo memo = {
        .epoch = layoutEpoch,
        .generation = generation,
        .available = available,
        .frame = frame,
        .key = LayoutKey::Of(ctx),
        .result = result,
    };
    if (measureMemos.size() < kMeasureMemos) {
        measureMemos.push_back(std::move(memo));
    } else {
        measureMemos[nextMeasureMemo] = std::move(memo);
        nextMeasureMemo = (nextMeasureMemo + 1) % kMeasureMemos;
    }
    return result;
}

bool Component::Impl::laidOut(const Context& ctx, Rect childFrame, Rect childClip, Extent2D<F32> childOffset) const {
    return placed && laidEpoch == layoutEpoch && !layoutDirty && !subtreeLayoutDirty && layoutKey.has_value() &&
           layoutKey->matches(ctx) &&
           frame == childFrame && clip == childClip && offset == childOffset;
}

void Component::Impl::runLayout(const Context& ctx) {
    if (!layoutKey.has_value() || !layoutKey->matches(ctx)) {
        layoutKey = LayoutKey::Of(ctx);
    }
    layoutDirty = false;
    subtreeLayoutDirty = false;
    laidEpoch = layoutEpoch;
    Component* const outer = layingOut;
    const U64 mark = pendingPopups.size();
    layingOut = self;
    layoutTime = ctx.time;
    if (!requestedFrames.empty()) {
        std::erase_if(requestedFrames, [this](const RequestedFrame& request) {
            return request.component == self;
        });
    }
    ++layoutRuns;
    self->layout(ctx);
    layingOut = outer;
    if (ctx.popups && (popupOwner || pendingPopups.size() > mark)) {
        std::vector<PopupRequest> requests;
        for (U64 index = mark; index < pendingPopups.size(); ++index) {
            requests.push_back(std::move(pendingPopups[index].request));
        }
        pendingPopups.resize(mark);
        popupOwner = !requests.empty();
        ctx.popups->replace(*self, alive, std::move(requests));
    }
    settleSubtree();
}

void Component::Impl::settleSubtree() {
    subtreeLayoutDirty = false;
    for (auto* child : children) {
        if (child && child->impl->visible && child->impl->placed &&
            (child->impl->layoutDirty || child->impl->subtreeLayoutDirty)) {
            subtreeLayoutDirty = true;
            break;
        }
    }
}

void Component::Impl::invalidateLayoutTree() {
    ++layoutEpoch;
}

void Component::layoutRoot(const Context& ctx, Rect frame) {
    const bool current = impl->laidOut(ctx, frame, frame, ctx.translation);
    impl->placed = true;
    impl->offset = ctx.translation;
    impl->setFrame(frame);
    impl->setClip(frame);
    if (!current) {
        impl->runLayout(ctx);
    }
}

void Component::layoutChild(const Context& ctx, Component& child, Rect frame) {
    impl->placeChild(ctx, child, frame, std::nullopt);
}

void Component::layoutChild(const Context& ctx, Component& child, Rect frame, Rect clip) {
    impl->placeChild(ctx, child, frame, clip);
}

void Component::Impl::placeChild(const Context& ctx, Component& child, Rect childFrame, std::optional<Rect> bounds) {
    const Extent2D<F32> base = translation();
    const Extent2D<F32> shift = {ctx.translation.x - base.x, ctx.translation.y - base.y};
    Rect childClipRect = childClip();
    childClipRect.x -= shift.x;
    childClipRect.y -= shift.y;
    if (bounds.has_value()) {
        childClipRect = Intersect(childClipRect, *bounds);
    }
    const bool current = child.impl->laidOut(ctx, childFrame, childClipRect, shift);
    child.impl->placed = true;
    if (child.impl->offset != shift) {
        child.impl->offset = shift;
        child.impl->invalidate(Dirty::Paint);
        child.impl->invalidatePaintTree();
    }
    child.impl->setFrame(childFrame);
    child.impl->setClip(childClipRect);
    if (!current) {
        child.impl->runLayout(ctx);
        if (layingOut != self) {
            settleSubtree();
        }
    }
}

void Component::layoutChildren(const Context& ctx) {
    impl->layoutChildren(ctx);
}

bool Component::mouseEventChildren(const MouseEvent& event) {
    if (event.type == MouseEventType::Cancel) {
        const std::vector<Component*> snapshot = impl->children;
        for (auto* child : snapshot) {
            if (child && hasChild(*child)) {
                child->mouseEvent(event);
            }
        }
        return false;
    }
    const U64 generation = impl->childGeneration;
    for (U64 index = impl->children.size(); index-- > 0;) {
        auto* child = impl->children[index];
        if (!child) {
            continue;
        }
        const bool handled = deliverChild(*child, event);
        if (handled || impl->childGeneration != generation) {
            return handled;
        }
    }
    return false;
}

bool Component::deliverChild(Component& child, const MouseEvent& event) {
    if (!hasChild(child) || !child.impl->visible) {
        return false;
    }
    const MouseEvent local = child.fromParent(event);
    if (local.type != MouseEventType::Move &&
        local.type != MouseEventType::Leave &&
        local.type != MouseEventType::Release &&
        local.type != MouseEventType::Cancel &&
        !child.impl->clip.contains(local.position.x, local.position.y)) {
        return false;
    }
    return child.mouseEvent(local);
}

}  // namespace Jetstream::Sakura
