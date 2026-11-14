#include "input_router.hh"

namespace Jetstream::Sakura {

InputRouter::Output InputRouter::route(const InputFrame& input, const Rect& screen) {
    Output output;
    const void* identity = input.source ? input.source : &input;
    if (source != identity || generation != input.sourceGeneration) {
        cancel(output);
        source = identity;
        generation = input.sourceGeneration;
        sampled = false;
        pointer.reset();
    }
    if (!std::isfinite(screen.x) || !std::isfinite(screen.y) ||
        !std::isfinite(screen.width) || !std::isfinite(screen.height) ||
        !std::isfinite(screen.x + screen.width) || !std::isfinite(screen.y + screen.height) ||
        screen.width <= 0.0f || screen.height <= 0.0f) {
        cancel(output);
        routed = input.serial;
        sampled = true;
        return state(output);
    }
    if (sampled && input.serial == routed) {
        return state(output);
    }
    routed = input.serial;
    sampled = true;
    if (!input.windowFocused && (captured() || scrolling || touchScrolling || !keys.empty())) {
        cancel(output);
    }

    const auto locate = [&](const Extent2D<F32>& point) {
        pointer = point;
        hovered = screen.contains(point.x, point.y);
        position = {(point.x - screen.x) / screen.width, (point.y - screen.y) / screen.height};
    };
    bool focused = true;
    bool pointerCancelled = false;
    const auto passive = [](const MouseEvent& mouse) {
        return mouse.type == MouseEventType::Move || mouse.type == MouseEventType::Scroll;
    };
    for (const auto& event : input.events) {
        if (const auto* focus = std::get_if<FocusEvent>(&event)) {
            focused = focus->focused;
            if (!focused) cancel(output);
            continue;
        }
        const bool engaged = focused && input.windowFocused;
        if (const auto* key = std::get_if<KeyEvent>(&event)) {
            if (engaged && track(*key)) output.events.push_back(*key);
            continue;
        }
        if (const auto* text = std::get_if<TextEvent>(&event)) {
            if (engaged) output.events.push_back(*text);
            continue;
        }
        const auto* mouse = std::get_if<MouseEvent>(&event);
        if (!mouse || (!engaged && !passive(*mouse))) continue;
        if (mouse->type == MouseEventType::Cancel ||
            (mouse->type == MouseEventType::Release && mouse->cancel)) {
            cancel(output, true);
            pointerCancelled = true;
            continue;
        }
        const bool edge = mouse->type == MouseEventType::Click || mouse->type == MouseEventType::Release;
        const U64 index = static_cast<U64>(mouse->button);
        if ((edge && index >= buttons.size()) ||
            (!edge && mouse->type != MouseEventType::Move && mouse->type != MouseEventType::Scroll &&
             mouse->type != MouseEventType::Zoom)) {
            continue;
        }
        if (!mouse->positioned) {
            if (mouse->type == MouseEventType::Click) {
                output.outsideClick |= mouse->button != MouseButton::Middle;
            } else if (mouse->type == MouseEventType::Release && buttons[index]) {
                cancel(output, true);
                pointerCancelled = true;
            }
            continue;
        } else if (!valid(mouse->position)) {
            cancel(output);
            pointerCancelled = true;
            continue;
        } else {
            locate(mouse->position);
            pointerCancelled = false;
        }
        MouseEvent routedEvent = *mouse;
        routedEvent.position = position;
        routedEvent.positioned = true;
        switch (mouse->type) {
            case MouseEventType::Click:
                if (!hovered) {
                    output.outsideClick |= mouse->button != MouseButton::Middle;
                    continue;
                }
                if (buttons[index]) continue;
                buttons[index] = true;
                break;
            case MouseEventType::Release:
                if (!buttons[index]) continue;
                buttons[index] = false;
                break;
            case MouseEventType::Scroll:
                if (!hovered && !(mouse->touch && touchScrolling) && mouse->phase != ScrollPhase::Ended &&
                    mouse->phase != ScrollPhase::MomentumEnded) continue;
                if (mouse->phase == ScrollPhase::None) {
                    break;
                }
                if (!mouse->touch) {
                    scrolling = mouse->phase != ScrollPhase::Ended && mouse->phase != ScrollPhase::MomentumEnded;
                    break;
                }
                if (mouse->cancel || mouse->phase == ScrollPhase::MomentumEnded) {
                    touchScrolling = false;
                } else if (mouse->phase == ScrollPhase::Began || mouse->phase == ScrollPhase::Changed) {
                    touchScrolling = true;
                }
                touchReleased = touchScrolling && mouse->phase == ScrollPhase::Ended;
                touchMomentum = touchScrolling && mouse->phase == ScrollPhase::Momentum;
                break;
            case MouseEventType::Move:
                if (!hovered && !captured() && !touchScrolling) continue;
                break;
            case MouseEventType::Zoom:
                if (!hovered || !std::isfinite(mouse->zoom) || mouse->zoom <= 0.0f) continue;
                break;
            default:
                continue;
        }
        output.events.push_back(routedEvent);
    }

    for (U64 index = 0; index < buttons.size(); ++index) {
        if (buttons[index] && !input.buttons[index]) {
            cancel(output);
            return state(output);
        }
    }
    if (!input.pointer || !valid(*input.pointer)) {
        if (touchReleased && !input.pointer) {
            touchScrolling = false;
            touchReleased = false;
        }
        const bool touchHeld = touchScrolling && !(touchMomentum && !input.pointer);
        if (captured() || scrolling || touchHeld) {
            cancel(output);
        }
        hovered = false;
        pointer.reset();
    } else if (!pointerCancelled) {
        const auto& held = input.modifiers;
        const bool shifted = held.control != modifiers.control || held.shift != modifiers.shift ||
                             held.alt != modifiers.alt || held.super != modifiers.super;
        const bool moved = !pointer || *pointer != *input.pointer;
        locate(*input.pointer);
        if ((moved || shifted) && (hovered || captured() || touchScrolling)) {
            output.events.push_back(MouseEvent{
                .type = MouseEventType::Move,
                .position = position,
                .modifiers = input.modifiers,
            });
        }
    } else {
        hovered = false;
    }
    modifiers = input.modifiers;
    return state(output);
}

void InputRouter::cancel(Output& output, bool explicitEvent) {
    const bool held = captured();
    for (const KeyCode key : keys) {
        output.events.push_back(KeyEvent{KeyEventType::Release, key, {}});
    }
    keys.clear();
    for (U64 index = 0; index < buttons.size(); ++index) {
        if (buttons[index]) {
            buttons[index] = false;
            output.events.push_back(MouseEvent{
                .type = MouseEventType::Release,
                .button = static_cast<MouseButton>(index),
                .position = position,
                .cancel = true,
            });
        }
    }
    if (held || scrolling || touchScrolling || explicitEvent) {
        output.events.push_back(MouseEvent{.type = MouseEventType::Cancel});
    }
    scrolling = false;
    touchScrolling = false;
    touchReleased = false;
    touchMomentum = false;
    hovered = false;
}

}  // namespace Jetstream::Sakura
