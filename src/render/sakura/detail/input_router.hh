#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_INPUT_ROUTER_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_INPUT_ROUTER_HH

#include "../input.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace Jetstream::Sakura {

struct InputRouter {
    struct Output {
        std::vector<FrameEvent> events;
        bool hovered = false;
        bool active = false;
        bool middle = false;
        bool outsideClick = false;
    };

    Output route(const InputFrame& input, const Rect& screen) {
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
            screen.width <= 0.0f || screen.height <= 0.0f || !input.windowFocused) {
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

        const auto locate = [&](const Extent2D<F32>& point) {
            pointer = point;
            hovered = screen.contains(point.x, point.y);
            position = {(point.x - screen.x) / screen.width, (point.y - screen.y) / screen.height};
        };
        bool focused = true;
        bool pointerCancelled = false;
        for (const auto& event : input.events) {
            if (const auto* focus = std::get_if<FocusEvent>(&event)) {
                focused = focus->focused;
                if (!focused) cancel(output);
                continue;
            }
            if (!focused) continue;
            if (const auto* key = std::get_if<KeyEvent>(&event)) {
                if (track(*key)) output.events.push_back(*key);
                continue;
            }
            if (const auto* text = std::get_if<TextEvent>(&event)) {
                output.events.push_back(*text);
                continue;
            }
            const auto* mouse = std::get_if<MouseEvent>(&event);
            if (!mouse) continue;
            if (mouse->type == MouseEventType::Cancel ||
                (mouse->type == MouseEventType::Release && mouse->cancel)) {
                cancel(output, true);
                pointerCancelled = true;
                continue;
            }
            const bool edge = mouse->type == MouseEventType::Click || mouse->type == MouseEventType::Release;
            const U64 index = static_cast<U64>(mouse->button);
            if ((edge && index >= buttons.size()) ||
                (!edge && mouse->type != MouseEventType::Move && mouse->type != MouseEventType::Scroll)) {
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
                    if (!hovered) continue;
                    break;
                case MouseEventType::Move:
                    if (!hovered && !captured()) continue;
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
            if (captured()) cancel(output);
            hovered = false;
            pointer.reset();
        } else if (focused && !pointerCancelled) {
            const bool moved = !pointer || *pointer != *input.pointer;
            locate(*input.pointer);
            if (moved && (hovered || captured())) {
                output.events.push_back(MouseEvent{
                    .type = MouseEventType::Move,
                    .position = position,
                    .modifiers = input.modifiers,
                });
            }
        } else {
            hovered = false;
        }
        return state(output);
    }

    void reset() {
        source = nullptr;
        generation = 0;
        sampled = false;
        hovered = false;
        pointer.reset();
        position = {0.0f, 0.0f};
        buttons.fill(false);
        keys.clear();
    }

 private:
    const void* source = nullptr;
    U64 generation = 0;
    U64 routed = 0;
    bool sampled = false;
    bool hovered = false;
    std::optional<Extent2D<F32>> pointer;
    Extent2D<F32> position = {0.0f, 0.0f};
    std::array<bool, 3> buttons{};
    std::vector<KeyCode> keys;

    static bool valid(const Extent2D<F32>& point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
    }

    bool captured() const {
        return std::any_of(buttons.begin(), buttons.end(), [](bool held) { return held; });
    }

    bool track(const KeyEvent& key) {
        const auto it = std::find(keys.begin(), keys.end(), key.key);
        if (key.type == KeyEventType::Press) {
            if (it == keys.end()) keys.push_back(key.key);
            return true;
        }
        if (it == keys.end()) return false;
        keys.erase(it);
        return true;
    }

    void cancel(Output& output, bool explicitEvent = false) {
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
        if (held || explicitEvent) output.events.push_back(MouseEvent{.type = MouseEventType::Cancel});
        hovered = false;
    }

    Output& state(Output& output) const {
        output.hovered = hovered;
        output.active = buttons[0] || buttons[1];
        output.middle = buttons[2];
        return output;
    }
};

}  // namespace Jetstream::Sakura

#endif
