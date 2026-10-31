#include "root_component.hh"

#include <jetstream/render/sakura/typography.hh>

#include <cmath>
#include <limits>

namespace Jetstream::Sakura {

void RootComponent::PreviewMotion::advance(F64 now, std::optional<Rect> goal) {
    if (goal.has_value() && *goal != to) {
        from = alpha > 0.0f ? rect : *goal;
        to = *goal;
        moveStart = now;
    }
    const F32 wanted = goal.has_value() ? 1.0f : 0.0f;
    if (wanted != alphaTo) {
        alphaFrom = alpha;
        alphaTo = wanted;
        fadeStart = now;
    }
    const F32 move = Progress(now, moveStart);
    if (move >= 1.0f) {
        rect = to;
    } else {
        const F32 t = Ease(move);
        rect = {Mix(from.x, to.x, t), Mix(from.y, to.y, t), Mix(from.width, to.width, t),
                Mix(from.height, to.height, t)};
    }
    const F32 fade = Progress(now, fadeStart);
    alpha = fade >= 1.0f ? alphaTo : Mix(alphaFrom, alphaTo, Ease(fade));
}

void RootComponent::setLayers(std::vector<Layer*> next) {
    layers = std::move(next);
    std::vector<Component*> children = {&background};
    for (auto* layer : layers) {
        children.push_back(layer);
    }
    children.push_back(&popups);
    children.push_back(&preview);
    children.push_back(&ghostFrame);
    children.push_back(&ghostIcon);
    children.push_back(&ghost);
    if (captured && std::find(children.begin(), children.end(), captured) == children.end()) {
        cancelCapture();
    }
    setChildren(children);
    prepareInput();
}

void RootComponent::cancelCapture(bool notify) {
    if (!captured && !drag.active) {
        return;
    }
    auto* previous = captured;
    const auto alive = capturedAlive;
    for (U64 index = 0; index < capturedButtons.size(); ++index) {
        if (notify) {
            cancelledButtons[index] = cancelledButtons[index] || capturedButtons[index];
        }
    }
    captured = nullptr;
    capturedTarget = nullptr;
    capturedButtons.fill(false);
    if (drag.active) {
        invalidate(Dirty::Layout);
    }
    drag.reset();
    if (notify && previous && !alive.expired()) {
        Access::mouseEvent(*previous, MouseEvent{.type = MouseEventType::Cancel});
    }
}

Component* RootComponent::activeModal() const {
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        if (Access::reachable(**it) && (*it)->modal()) {
            return *it;
        }
    }
    return nullptr;
}

void RootComponent::prepareInput() {
    if (preparingInput) {
        return;
    }
    preparingInput = true;
    const auto token = Access::alive(*this);
    struct Guard {
        RootComponent* root;
        std::weak_ptr<bool> alive;
        ~Guard() { if (!alive.expired()) root->preparingInput = false; }
    } guard{this, token};
    popups.prepareInput();
    if (token.expired()) {
        return;
    }
    if (captured && (capturedAlive.expired() || !Access::attached(*captured, *this) ||
        !Access::reachable(*captured) || (captured == &popups ? !popups.capturesInput() : !acceptsInput(*captured)) ||
        (capturedTarget && (capturedTargetAlive.expired() || !Access::attached(*capturedTarget, *this) ||
                           !Access::reachable(*capturedTarget) || !acceptsInput(*capturedTarget))))) {
        cancelCapture();
        if (token.expired()) {
            return;
        }
    }
    auto* focused = Access::focusedInTree(*this);
    for (auto& entry : modalFocus) {
        if (entry.modal == keyboardModal && !entry.alive.expired() && focused && inModal(*focused, *entry.modal)) {
            entry.focused = focused;
            entry.focusedAlive = Access::alive(*focused);
        }
    }
    Component* restore = nullptr;
    std::weak_ptr<bool> restoreAlive;
    for (U64 index = 0; index < modalFocus.size();) {
        const auto entry = modalFocus[index];
        if (!entry.alive.expired() && Access::attached(*entry.modal, *this) &&
            Access::reachable(*entry.modal) && static_cast<Layer*>(entry.modal)->modal()) {
            ++index;
            continue;
        }
        if (entry.modal == keyboardModal) {
            restore = entry.previous;
            restoreAlive = entry.previousAlive;
        }
        for (U64 above = index + 1; above < modalFocus.size(); ++above) {
            auto& next = modalFocus[above];
            if (next.previous && (next.previousAlive.expired() ||
                (!entry.alive.expired() && inModal(*next.previous, *entry.modal)))) {
                next.previous = entry.previous;
                next.previousAlive = entry.previousAlive;
            }
        }
        modalFocus.erase(modalFocus.begin() + static_cast<I64>(index));
    }
    auto* modal = activeModal();
    const bool changed = keyboardModal != modal;
    keyboardModal = modal;
    if (changed) {
        Access::invalidateLayout(popups);
    }
    auto entry = std::find_if(modalFocus.begin(), modalFocus.end(), [&](const auto& state) {
        return state.modal == modal;
    });
    if (modal && entry == modalFocus.end()) {
        const bool inherited = restore && !restoreAlive.expired();
        auto* previous = inherited ? restore : focused;
        if (previous && inModal(*previous, *modal)) {
            previous = nullptr;
        }
        modalFocus.push_back({.modal = modal, .alive = Access::alive(*modal), .previous = previous,
                              .previousAlive = previous ? Access::alive(*previous) : std::weak_ptr<bool>{}});
        entry = modalFocus.end() - 1;
    }
    if (focused && !acceptsInput(*focused)) {
        Access::blur(*focused);
        if (token.expired()) {
            return;
        }
    }
    if (Access::focusedInTree(*this)) {
        pendingFocus = nullptr;
        return;
    }
    if (!changed) {
        restore = pendingFocus;
        restoreAlive = pendingFocusAlive;
    } else if (entry != modalFocus.end() && entry->focused && !entry->focusedAlive.expired()) {
        restore = entry->focused;
        restoreAlive = entry->focusedAlive;
    }
    pendingFocus = nullptr;
    if (restore && !restoreAlive.expired() && Access::attached(*restore, *this) &&
        acceptsInput(*restore)) {
        if (Access::reachable(*restore)) {
            Access::focus(*restore);
        } else if (Access::within(*restore, popups)) {
            pendingFocus = restore;
            pendingFocusAlive = restoreAlive;
        }
    }
}

void RootComponent::layout(const Context& ctx) {
    const auto token = Access::alive(*this);
    prepareInput();
    if (token.expired()) {
        return;
    }
    const Rect bounds = frame();
    background.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color("background")}},
    });
    layoutChild(ctx, background, bounds);

    Context layerCtx = ctx;
    layerCtx.popups = &popups;
    layerCtx.dockDrag = &drag;
    if (drag.active != dragged) {
        dragged = drag.active;
        relayoutLayers();
    }
    const bool offering = reoffer;
    reoffer = false;
    if (offering) {
        drag.begin();
    }
    for (auto* layer : layers) {
        layoutChild(layerCtx, *layer, bounds);
    }
    layoutChild(layerCtx, popups, bounds);
    prepareInput();
    if (token.expired()) {
        return;
    }
    if (offering) {
        drag.commit();
    }

    const bool dragging = drag.active && ctx.mouse.has_value();
    std::optional<Rect> goal;
    if (dragging && drag.target.has_value()) {
        goal = drag.target->preview;
    }
    motion.advance(ctx.time, goal);
    if (!motion.settled()) {
        requestFrame();
    }
    auto fill = ctx.color("drop_fill");
    fill.a *= motion.alpha;
    auto border = ctx.color("drop_border");
    border.a *= motion.alpha;
    preview.update({
        .instances = {{.rect = motion.rect, .visible = motion.alpha > 0.0f, .backgroundColor = fill}},
        .cornerRadius = kPreviewRadius * ctx.pixelRatio,
        .borderWidth = kPreviewBorder * ctx.pixelRatio,
        .borderColor = border,
    });
    layoutChild(ctx, preview, bounds);

    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 font = Typography::FontSize * ctx.pixelRatio;
    ghostIcon.update({.str = drag.icon, .colorKey = "pill_text"});
    ghost.update({.str = drag.title, .colorKey = "pill_text"});
    const F32 iconWidth = drag.icon.empty() ? 0.0f : measureChild(ghostIcon, ctx, {infinite, infinite}).x;
    const F32 titleWidth = measureChild(ghost, ctx, {infinite, infinite}).x;
    const F32 padX = kGhostPadX * ctx.pixelRatio;
    const F32 height = std::round(std::max(kGhostHeight * ctx.pixelRatio, font + 4.0f * ctx.pixelRatio));
    const F32 iconGap = iconWidth > 0.0f ? kGhostIconGap * ctx.pixelRatio : 0.0f;
    const F32 width = std::ceil(2.0f * padX + iconWidth + iconGap + titleWidth);
    const Extent2D<F32> mouse = ctx.mouse.value_or(Extent2D<F32>{0.0f, 0.0f});
    const F32 offset = kGhostOffset * ctx.pixelRatio;
    const Rect ghostRect = dragging ? Rect{mouse.x + offset, mouse.y + offset, width, height} : Rect{};
    ghostFrame.update({
        .instances = {{.rect = ghostRect, .visible = dragging, .backgroundColor = ctx.color("pill_active_bg")}},
        .cornerRadius = height * 0.5f,
        .borderWidth = ctx.pixelRatio,
        .borderColor = ctx.color("tile_border"),
        .strokeFringe = true,
    });
    layoutChild(ctx, ghostFrame, bounds);
    const F32 textY = ghostRect.y + std::floor((height - font) * 0.5f);
    ghostIcon.setVisible(dragging && iconWidth > 0.0f);
    layoutChild(ctx, ghostIcon, {ghostRect.x + padX, textY, iconWidth, font});
    ghost.setVisible(dragging);
    layoutChild(ctx, ghost, {ghostRect.x + padX + iconWidth + iconGap, textY, titleWidth, font});
}

void RootComponent::relayoutLayers() {
    reoffer = true;
    for (auto* layer : layers) {
        Access::invalidateLayout(*layer);
    }
}

Component* RootComponent::target(const Extent2D<F32>& point) {
    if (popups.covers(point)) {
        return &popups;
    }
    for (U64 index = layers.size(); index-- > 0;) {
        if (Access::reachable(*layers[index]) && acceptsInput(*layers[index]) &&
            (index == 0 || layers[index]->covers(point))) {
            return layers[index];
        }
    }
    return nullptr;
}

std::vector<Component*> RootComponent::ordered() {
    std::vector<Component*> result = {&popups};
    for (U64 index = layers.size(); index-- > 0;) {
        result.push_back(layers[index]);
    }
    return result;
}

bool RootComponent::mouseEvent(const MouseEvent& event) {
    const auto token = Access::alive(*this);
    prepareInput();
    if (token.expired()) {
        return true;
    }
    if (event.type == MouseEventType::Leave || event.type == MouseEventType::Cancel ||
        (event.type == MouseEventType::Release && event.cancel)) {
        auto cancelled = event;
        if (event.cancel) {
            cancelled.type = MouseEventType::Cancel;
        }
        cancelCapture(false);
        for (auto* layer : ordered()) {
            deliver(*layer, cancelled);
            if (token.expired()) {
                return true;
            }
        }
        return false;
    }

    const U64 button = static_cast<U64>(event.button);
    if ((event.type == MouseEventType::Click || event.type == MouseEventType::Release) &&
        button >= capturedButtons.size()) {
        return false;
    }
    if (event.type == MouseEventType::Release && cancelledButtons[button]) {
        cancelledButtons[button] = false;
        return true;
    }
    Component* focus = captured ? captured : target(event.position);
    if (event.type == MouseEventType::Click) {
        cancelledButtons[button] = false;
        if (!captured && focus) {
            capturedAlive = Access::alive(*focus);
            capturedTarget = focus == &popups ? nullptr : Access::pointerTarget(*this, event.position);
            capturedTargetAlive = capturedTarget ? Access::alive(*capturedTarget) : std::weak_ptr<bool>{};
        }
        captured = focus;
        capturedButtons[button] = focus != nullptr;
    }

    bool handled = false;
    for (auto* layer : ordered()) {
        if (layer == focus) {
            handled = deliver(*layer, event) || handled;
        } else if (event.type == MouseEventType::Move) {
            deliver(*layer, away(event));
        }
        if (token.expired()) {
            return true;
        }
    }
    if (event.type == MouseEventType::Release) {
        if (event.button == MouseButton::Left) {
            drag.reset();
        }
        capturedButtons[button] = false;
        if (std::none_of(capturedButtons.begin(), capturedButtons.end(), [](bool held) { return held; })) {
            captured = nullptr;
        }
    }
    if (drag.active && dragPoint != event.position) {
        dragPoint = event.position;
        invalidate(Dirty::Layout);
        relayoutLayers();
    }
    return handled;
}

}  // namespace Jetstream::Sakura
