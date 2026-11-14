#include "touch_gesture.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Viewport {

TouchGesture::TouchGesture(Render::InputQueue& input) : input(input) {}

bool TouchGesture::active() const {
    return !contacts.empty() && action != Action::Ignored;
}

void TouchGesture::resumePointer() {
    if (!active()) {
        stop();
        pointerOwned = false;
    }
}

void TouchGesture::scroll(U32 timestamp, ScrollPhase phase, Extent2D<F32> delta, bool cancelled) {
    constexpr F32 pointsPerScrollUnit = 10.0f;
    const Extent2D<F32> units = {delta.x / pointsPerScrollUnit, delta.y / pointsPerScrollUnit};
    if (phase == ScrollPhase::Began || phase == ScrollPhase::Changed) {
        momentum.sample(timestamp, units);
    }
    input.push(Render::InputQueue::Wheel{
        .delta = units,
        .phase = phase,
        .precise = true,
        .position = origin,
        .touch = true,
        .cancel = cancelled,
    });
}

void TouchGesture::release(U32 timestamp, F64 now) {
    constexpr F64 touchDecay = 4.5;
    scroll(timestamp, ScrollPhase::Ended);
    coasting = momentumAllowed && momentum.release(timestamp, touchDecay);
    if (coasting) {
        steppedAt = now;
        scroll(timestamp, ScrollPhase::Momentum);
    } else {
        momentum.reset();
    }
}

void TouchGesture::stop() {
    if (coasting) {
        scroll(0, ScrollPhase::MomentumEnded, {}, true);
    }
    coasting = false;
    momentum.reset();
}

void TouchGesture::handle(const Event& event, F64 now, bool allowStart) {
    using Type = Event::Type;
    if (event.type == Type::Cancel) {
        cancel();
        return;
    }
    if (event.type == Type::Frame) {
        if (active()) {
            momentumAllowed = momentumAllowed && event.momentumAllowed;
        }
        pinchFrame();
        return;
    }

    const auto contact = std::find_if(contacts.begin(), contacts.end(),
                                      [&](const auto& contact) { return contact.id == event.id; });
    if (event.type == Type::Down) {
        if (contact != contacts.end()) {
            return;
        }
        const bool first = contacts.empty();
        contacts.push_back({event.id, event.position});
        if (first) {
            stop();
            action = allowStart ? Action::Pending : Action::Ignored;
            momentumAllowed = event.momentumAllowed;
            if (!allowStart) {
                return;
            }
            pointerOwned = true;
            origin = event.position;
            last = event.position;
            startedAt = now;
            startedTimestamp = event.timestamp;
            input.push(Render::InputQueue::Pointer{event.position});
        } else if (action != Action::Ignored) {
            momentumAllowed = momentumAllowed && event.momentumAllowed;
            if (contacts.size() == 2 && action != Action::Pinch) {
                if (action == Action::Swipe) {
                    scroll(event.timestamp, ScrollPhase::Ended, {}, true);
                }
                action = Action::Pinch;
                pinchActive = true;
                pinchDistance = contactDistance();
                const auto& firstPosition = contacts.front().position;
                pinchPosition = {(firstPosition.x + event.position.x) * 0.5f,
                                 (firstPosition.y + event.position.y) * 0.5f};
                input.push(Render::InputQueue::Pointer{pinchPosition});
            }
        }
        return;
    }
    if (contact == contacts.end()) {
        return;
    }
    momentumAllowed = momentumAllowed && event.momentumAllowed;

    if (event.type == Type::Motion) {
        contact->position = event.position;
        if (pinchActive && contact - contacts.begin() < 2) {
            pinchDirty = true;
            return;
        }
        if (action == Action::Ignored || action == Action::ContextMenu || action == Action::Pinch) {
            return;
        }
        if (action == Action::Pending) {
            constexpr F32 dragDistance = 8.0f;
            const F32 dx = event.position.x - origin.x;
            const F32 dy = event.position.y - origin.y;
            if (dx * dx + dy * dy <= dragDistance * dragDistance) {
                return;
            }
            action = Action::Swipe;
            scroll(event.timestamp, ScrollPhase::Began, {dx, dy});
        } else {
            scroll(event.timestamp, ScrollPhase::Changed,
                   {event.position.x - last.x, event.position.y - last.y});
        }
        input.push(Render::InputQueue::Pointer{event.position});
        last = event.position;
        return;
    }
    if (event.type == Type::Up) {
        if (pinchActive && contact - contacts.begin() < 2) {
            pinchFrame();
            pinchActive = false;
        }
        if (contacts.size() == 1) {
            hold(static_cast<U32>(event.timestamp - startedTimestamp) / 1000.0);
        }
        contacts.erase(contact);
        if (!contacts.empty()) {
            return;
        }
        if (action != Action::Ignored) {
            if (action == Action::Pending) {
                input.push(Render::InputQueue::Button{MouseButton::Left, true});
                input.push(Render::InputQueue::Button{MouseButton::Left, false});
            } else if (action == Action::Swipe) {
                release(event.timestamp, now);
            }
            input.push(Render::InputQueue::Pointer{std::nullopt});
        }
        action = Action::Pending;
        pinchDirty = false;
    }
}

void TouchGesture::poll(F64 now) {
    hold(now - startedAt);
    if (!coasting) {
        return;
    }
    const Extent2D<F32> delta = momentum.advance(now - steppedAt);
    steppedAt = now;
    if (delta.x != 0.0f || delta.y != 0.0f) {
        input.push(Render::InputQueue::Wheel{
            .delta = delta,
            .phase = ScrollPhase::Momentum,
            .precise = true,
            .position = origin,
            .touch = true,
        });
    }
    if (!momentum.active()) {
        coasting = false;
        scroll(0, ScrollPhase::MomentumEnded);
    }
}

void TouchGesture::hold(F64 duration) {
    constexpr F64 holdDuration = 0.5;
    if (!active() || action != Action::Pending || duration < holdDuration) {
        return;
    }
    action = Action::ContextMenu;
    input.push(Render::InputQueue::Pointer{origin});
    input.push(Render::InputQueue::Button{MouseButton::Right, true});
    input.push(Render::InputQueue::Button{MouseButton::Right, false});
}

void TouchGesture::cancel() {
    if (active()) {
        if (action == Action::Swipe) {
            scroll(0, ScrollPhase::Ended, {}, true);
        }
        input.push(Render::InputQueue::Pointer{std::nullopt});
    }
    stop();
    contacts.clear();
    action = Action::Pending;
    pinchActive = false;
    pinchDirty = false;
}

F32 TouchGesture::contactDistance() const {
    return std::hypot(contacts[1].position.x - contacts[0].position.x,
                      contacts[1].position.y - contacts[0].position.y);
}

void TouchGesture::pinchFrame() {
    if (!pinchActive || !pinchDirty) {
        return;
    }
    pinchDirty = false;
    const F32 distance = contactDistance();
    constexpr F32 minimumDistance = 8.0f;
    if (distance >= minimumDistance && pinchDistance >= minimumDistance && distance != pinchDistance) {
        input.push(Render::InputQueue::Zoom{pinchPosition, distance / pinchDistance});
    }
    pinchDistance = distance;
}

}  // namespace Jetstream::Viewport
