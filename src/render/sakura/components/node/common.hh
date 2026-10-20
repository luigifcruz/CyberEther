#ifndef JETSTREAM_RENDER_SAKURA_NODE_COMMON_HH
#define JETSTREAM_RENDER_SAKURA_NODE_COMMON_HH

#include <jetstream/render/sakura/components/node/content_stack.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../../context.hh"
#include "../../metrics.hh"
#include "../../detail/helpers.hh"
#include "../../detail/popup_layer.hh"
#include "../../detail/text_metrics.hh"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>

namespace Jetstream::Sakura::NodeStyle {

constexpr F32 BaseFont = Typography::FontSize;
constexpr F32 NodeFont = Typography::FontSize * Typography::ChromeScale;
constexpr F32 LabelFont = Typography::FontSize * 0.75f;
constexpr F32 FramePadX = 6.0f;
constexpr F32 FramePadY = 3.0f;
constexpr F32 FrameRounding = kFrameRounding;
constexpr F32 ItemSpacing = NodeContentStack::Spacing;
constexpr F32 UnitInset = 3.0f;
constexpr F32 StackedGap = 3.0f;
constexpr F32 CardLabelOffsetX = 6.0f;
constexpr F32 CardLabelOffsetY = 2.0f;
constexpr F32 FieldColumnMinWidth = 140.0f;
constexpr F32 TooltipWidth = 960.0f;
constexpr const char* MonoFont = Typography::MonoFont;
constexpr const char* BoldFont = "default_mono_bold";
constexpr const char* HeaderFont = "default_display_semibold";
constexpr const char* IconFont = Typography::IconFont;

inline F32 FontOf(const Context& ctx) {
    return ctx.fontSize > 0.0f ? ctx.fontSize : BaseFont;
}

inline F32 FrameHeight(const Context& ctx) {
    return FontOf(ctx) + 2.0f * FramePadY;
}

inline F32 CardLabelHeight(const Context& ctx) {
    return FontOf(ctx) + 14.0f;
}

inline Rect Snap(Rect rect) {
    const F32 x0 = std::round(rect.x);
    const F32 y0 = std::round(rect.y);
    const F32 x1 = std::round(rect.x + rect.width);
    const F32 y1 = std::round(rect.y + rect.height);
    return {x0, y0, std::max(0.0f, x1 - x0), std::max(0.0f, y1 - y0)};
}

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

inline std::string Ellipsize(TextMetrics& metrics,
                             const std::string& fontName,
                             const std::string& text,
                             F32 fontSize,
                             F32 width) {
    return metrics.ellipsize(fontName, text, fontSize, width);
}

inline std::string Trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

inline bool DecimalCharacter(U32 character) {
    return (character >= '0' && character <= '9') || character == '.' || character == '+' || character == '-';
}

inline bool ScientificCharacter(U32 character) {
    return DecimalCharacter(character) || character == 'e' || character == 'E';
}

inline std::optional<F64> ParseFloat(const std::string& text) {
    const std::string trimmed = Trim(text);
    if (trimmed.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const F64 value = std::strtod(trimmed.c_str(), &end);
    if (end == trimmed.c_str() || errno == ERANGE || !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

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

}  // namespace Jetstream::Sakura::NodeStyle

#endif  // JETSTREAM_RENDER_SAKURA_NODE_COMMON_HH
