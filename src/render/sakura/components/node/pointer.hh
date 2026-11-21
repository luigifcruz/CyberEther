#ifndef JETSTREAM_RENDER_SAKURA_NODE_POINTER_HH
#define JETSTREAM_RENDER_SAKURA_NODE_POINTER_HH

#include "../../context.hh"
#include "../../detail/popup_layer.hh"

#include <initializer_list>
#include <optional>

namespace Jetstream::Sakura {

constexpr F32 TooltipWidth = 960.0f;

inline PopupRequest TooltipRequest(const Context& ctx, const Rect& anchor, Component* content) {
    return {
        .content = content,
        .anchor = ctx.mouse.has_value() ? Rect{ctx.mouse->x, ctx.mouse->y, 1.0f, 1.0f} : anchor,
        .placement = PopupPlacement::Tooltip,
        .panel = true,
        .maxWidth = TooltipWidth,
        .interactive = false,
        .delayed = true,
    };
}

struct Pointer {
    std::optional<Extent2D<F32>> position;
    U64 hovered = 0;
    bool following = false;

    bool track(const MouseEvent& event, const Rect& area, std::initializer_list<Rect> regions) {
        if (event.type != MouseEventType::Move && event.type != MouseEventType::Leave &&
            event.type != MouseEventType::Cancel) {
            return false;
        }
        const auto previous = position;
        position.reset();
        if (event.type == MouseEventType::Move && area.contains(event.position.x, event.position.y)) {
            position = event.position;
        }
        U64 next = 0;
        U64 bit = 1;
        for (const auto& region : regions) {
            if (inside(region)) {
                next |= bit;
            }
            bit <<= 1;
        }
        const bool changed = next != hovered || (following && next != 0 && position != previous);
        hovered = next;
        return changed;
    }

    bool tooltip(const Context& ctx, const Rect& anchor, Component* content) {
        following = true;
        if (!content) {
            return false;
        }
        RequestPopup(ctx, TooltipRequest(ctx, anchor, content));
        return true;
    }

    bool inside(const Rect& rect) const {
        return position.has_value() && rect.contains(position->x, position->y);
    }
};

struct Pressable {
    bool pressed = false;

    bool click(const MouseEvent& event, const Rect& rect) {
        if (event.type == MouseEventType::Click && event.button == MouseButton::Left &&
            rect.contains(event.position.x, event.position.y)) {
            pressed = true;
            return true;
        }
        return false;
    }

    bool release(const MouseEvent& event, const Rect& rect, bool requireInside = true) {
        if (event.type == MouseEventType::Cancel || event.type == MouseEventType::Leave) {
            pressed = false;
            return false;
        }
        if (event.type != MouseEventType::Release || event.button != MouseButton::Left || !pressed) {
            return false;
        }
        pressed = false;
        return !requireInside || rect.contains(event.position.x, event.position.y);
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_POINTER_HH
