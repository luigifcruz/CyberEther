#include "touch_scroll.hh"
#include "access.hh"

#include <utility>

namespace Jetstream::Sakura {

struct TouchScroll::Capture {
    Capture* previous = capture;
    std::optional<Owner> owner;

    Capture() { capture = this; }
    ~Capture() { capture = previous; }
};

thread_local TouchScroll::Capture* TouchScroll::capture = nullptr;

struct TouchScroll::Delivery {
    Delivery* previous = delivery;
    Component* component;
    bool redirected = false;

    explicit Delivery(Component* component) : component(component) { delivery = this; }
    ~Delivery() { delivery = previous; }
};

thread_local TouchScroll::Delivery* TouchScroll::delivery = nullptr;

bool TouchScroll::dispatch(Component& component, const MouseEvent& event, Handler handler) {
    if (delivery && !capture && event.type == MouseEventType::Scroll && event.touch &&
        delivery->component != &component) {
        delivery->redirected |= event.scroll.x != 0.0f || event.scroll.y != 0.0f;
        return true;
    }
    if (!capture || event.type != MouseEventType::Scroll || !event.touch || event.phase != ScrollPhase::Began) {
        return handler(component, event);
    }
    Owner candidate{&component, handler, event.position, {}};
    for (auto* node = &component; node; node = Access::parent(*node)) {
        candidate.path.push_back({node, Access::alive(*node), Access::attachmentGeneration(*node),
                                  Access::hiddenGeneration(*node)});
    }
    auto* pending = capture;
    auto previous = pending->owner;
    const bool handled = handler(component, event);
    if (!handled) {
        pending->owner = std::move(previous);
    } else if (!pending->owner) {
        pending->owner = std::move(candidate);
    }
    return handled;
}

bool TouchScroll::begin(Component& next, const MouseEvent& event) {
    const auto alive = Access::alive(next);
    const U64 previous = generation;
    reset();
    if (alive.expired() || generation != previous + 1) {
        return true;
    }
    root = &next;
    rootAlive = alive;
    const U64 started = generation;
    Capture pending;
    const bool handled = Access::mouseEvent(next, event);
    if (started != generation) {
        if (pending.owner) {
            cancel(*pending.owner);
        }
        return true;
    }
    owner = std::move(pending.owner);
    if (owner && !valid()) {
        reset();
    }
    return handled;
}

bool TouchScroll::valid() const {
    if (!owner || rootAlive.expired()) {
        return false;
    }
    Component* parent = nullptr;
    for (auto it = owner->path.rbegin(); it != owner->path.rend(); ++it) {
        if (it->alive.expired() || Access::parent(*it->component) != parent ||
            Access::attachmentGeneration(*it->component) != it->attachment ||
            Access::hiddenGeneration(*it->component) != it->hidden) {
            return false;
        }
        parent = it->component;
    }
    return Access::within(*owner->component, root) && Access::reachable(*owner->component) &&
           Access::acceptsInput(*root, *owner->component);
}

void TouchScroll::cancel(const Owner& target) {
    if (!target.path.front().alive.expired()) {
        Access::mouseEvent(*target.component, MouseEvent{.type = MouseEventType::Cancel});
    }
}

void TouchScroll::reset() {
    ++generation;
    const auto previous = std::exchange(owner, std::nullopt);
    root = nullptr;
    rootAlive.reset();
    if (previous && !previous->ended) {
        cancel(*previous);
    }
}

void TouchScroll::send(const MouseEvent& event) {
    if (!valid() || event.cancel) {
        reset();
        return;
    }
    const auto target = *owner;
    owner->ended = event.phase == ScrollPhase::Ended;
    auto local = event;
    local.position = target.position;
    const U64 started = generation;
    if (event.phase == ScrollPhase::Ended || event.phase == ScrollPhase::MomentumEnded) {
        Access::scrollPhaseEnded(*target.component, event.phase);
    }
    if (started != generation) {
        return;
    }
    if (!valid()) {
        reset();
        return;
    }
    Delivery routed(target.component);
    target.handler(*target.component, local);
    if (started != generation) {
        return;
    }
    if (routed.redirected || !valid()) {
        reset();
    } else if (event.phase == ScrollPhase::MomentumEnded) {
        owner.reset();
    }
}

}  // namespace Jetstream::Sakura
