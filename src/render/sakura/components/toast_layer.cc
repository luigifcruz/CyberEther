#include <jetstream/render/sakura/components/toast_layer.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../toast_queue.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kToastMargin = 20.0f;
constexpr F32 kToastSpacing = 10.0f;
constexpr F64 kToastFade = 0.150;
constexpr F32 kToastPadding = 16.0f;
constexpr F32 kToastIcon = 22.0f;
constexpr F32 kToastGap = 14.0f;
constexpr F32 kToastTitle = 16.0f;
constexpr F32 kToastLineGap = 6.0f;
constexpr F32 kToastTextWidth = 340.0f;
constexpr F32 kToastClose = 22.0f;
constexpr F32 kToastCloseGlyph = 11.0f;
constexpr F32 kToastSlide = 16.0f;
constexpr F32 kToastShadowSoftness = 8.0f;
constexpr F32 kToastShadowDrop = 3.0f;

}  // namespace

struct ToastLayer::Impl {
    struct Item {
        ToastType type = ToastType::Info;
        F64 duration = 3.0;
        F64 start = 0.0;
        F64 seen = 0.0;
        bool started = false;
        bool expired = false;
        bool hovered = false;
        bool held = false;
        bool closing = false;
        bool closeHovered = false;
        std::string title;
        std::string message;
        Rect card;
        Rect close;
        Card chrome;
        Text icon;
        Text heading;
        Text content;
        Box closeBadge;
        Text closeGlyph;

        std::vector<Component*> parts() {
            std::vector<Component*> result = {&chrome};
            result.push_back(&icon);
            result.push_back(&heading);
            if (!message.empty()) {
                result.push_back(&content);
            }
            result.push_back(&closeBadge);
            result.push_back(&closeGlyph);
            return result;
        }

        bool live() const {
            return started && !expired;
        }

        void dismiss(F64 now) {
            if (!started || expired) {
                return;
            }
            const F64 elapsed = now - start;
            if (elapsed < kToastFade + duration) {
                duration = std::max(0.0, elapsed - kToastFade);
            }
            held = false;
        }
    };

    std::vector<std::unique_ptr<Item>> items;
    Item* pressedClose = nullptr;

    Item* at(const Extent2D<F32>& point) const {
        for (auto it = items.rbegin(); it != items.rend(); ++it) {
            auto& item = **it;
            if (item.live() && (item.card.contains(point.x, point.y) || item.close.contains(point.x, point.y))) {
                return &item;
            }
        }
        return nullptr;
    }
};

ToastLayer::ToastLayer() {
    impl = std::make_unique<Impl>();
}

ToastLayer::~ToastLayer() = default;

bool ToastLayer::covers(const Extent2D<F32>& point) const {
    return impl->at(point) != nullptr;
}

void ToastLayer::update() {
    const U64 before = impl->items.size();
    std::erase_if(impl->items, [&](const auto& item) {
        if (item->expired && item.get() == impl->pressedClose) {
            impl->pressedClose = nullptr;
        }
        return item->expired;
    });
    for (auto& queued : Private::TakeToasts()) {
        auto item = std::make_unique<Impl::Item>();
        item->type = queued.type;
        item->duration = static_cast<F64>(std::max(0, queued.durationMs)) / 1000.0;
        item->message = queued.message;
        item->title = queued.title;
        impl->items.push_back(std::move(item));
    }
    if (impl->items.size() != before) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> children;
    for (auto& item : impl->items) {
        for (auto* part : item->parts()) {
            children.push_back(part);
        }
    }
    mountChildren(children);
}

void ToastLayer::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 padding = Px(ctx, kToastPadding);
    const F32 icon = Px(ctx, kToastIcon);
    const F32 gap = Px(ctx, kToastGap);
    const F32 badge = Px(ctx, kToastClose);
    const F32 line = Px(ctx, 1.0f);
    const F32 radius = Px(ctx, kPanelRounding);
    const F32 softness = Px(ctx, kToastShadowSoftness);
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 textLimit = std::max(Px(ctx, 120.0f), std::min(Px(ctx, kToastTextWidth), bounds.width * 0.5f));
    F32 bottom = bounds.bottom() - Px(ctx, kToastMargin);

    for (auto& itemPtr : impl->items) {
        auto& item = *itemPtr;
        if (!item.started) {
            item.started = true;
            item.start = ctx.time;
            item.seen = ctx.time;
        }
        const F64 holdStart = item.start + kToastFade;
        const bool holding = item.seen >= holdStart && item.seen <= holdStart + item.duration;
        if (item.held && holding) {
            item.start += ctx.time - item.seen;
        }
        item.held = item.hovered;
        item.seen = ctx.time;
        if (item.closing) {
            item.closing = false;
            item.dismiss(ctx.time);
        }
        if (item.expired || ctx.time - item.start > item.duration + 2.0 * kToastFade) {
            item.expired = true;
            item.card = {};
            item.close = {};
            for (auto* part : item.parts()) {
                part->setVisible(false);
            }
            continue;
        }
        const F64 elapsed = ctx.time - item.start;
        if (elapsed < kToastFade || elapsed > kToastFade + item.duration) {
            requestFrame();
        } else if (!item.hovered) {
            requestFrame(item.start + kToastFade + item.duration);
        }
        F32 opacity = 1.0f;
        if (elapsed < kToastFade) {
            opacity = static_cast<F32>(elapsed / kToastFade);
        } else if (elapsed > kToastFade + item.duration) {
            opacity = static_cast<F32>(1.0 - (elapsed - kToastFade - item.duration) / kToastFade);
        }
        opacity = std::clamp(opacity, 0.0f, 1.0f);
        const F32 eased = 1.0f - (1.0f - opacity) * (1.0f - opacity);
        const F32 slide = std::round((1.0f - eased) * Px(ctx, kToastSlide));

        const char* iconStr = ICON_LC_INFO;
        const char* fallback = "Info";
        const char* colorKey = "info_blue";
        switch (item.type) {
            case ToastType::Success:
                iconStr = ICON_LC_CIRCLE_CHECK;
                fallback = "Success";
                colorKey = "success_green";
                break;
            case ToastType::Warning:
                iconStr = ICON_LC_TRIANGLE_ALERT;
                fallback = "Warning";
                colorKey = "warning_yellow";
                break;
            case ToastType::Error:
                iconStr = ICON_LC_CIRCLE_ALERT;
                fallback = "Error";
                colorKey = "error_red";
                break;
            case ToastType::Info:
                break;
        }
        item.icon.update({.str = iconStr, .colorKey = colorKey, .scale = kToastIcon / Typography::FontSize,
                          .opacity = opacity});
        item.heading.update({.str = item.title.empty() ? fallback : item.title,
                             .font = Text::Font::DisplaySemiBold,
                             .colorKey = "text_primary",
                             .scale = kToastTitle / Typography::FontSize,
                             .opacity = opacity});
        if (!item.message.empty()) {
            item.content.update({.str = item.message, .tone = Text::Tone::Secondary, .wrapped = true,
                                 .maxCharacters = 4096, .opacity = opacity});
        }

        const Extent2D<F32> iconSize = measureChild(item.icon, ctx, {infinite, infinite});
        const Extent2D<F32> headingSize = measureChild(item.heading, ctx, {infinite, infinite});
        const F32 headingWidth = std::min(textLimit, headingSize.x);
        const F32 titleSize = headingSize.y;
        F32 textWidth = headingWidth;
        Extent2D<F32> contentSize = {0.0f, 0.0f};
        if (!item.message.empty()) {
            const F32 natural = measureChild(item.content, ctx, {infinite, infinite}).x;
            textWidth = std::max(textWidth, std::min(std::ceil(natural), textLimit));
            contentSize = measureChild(item.content, ctx, {textWidth, infinite});
        }
        const F32 textHeight = titleSize + (item.message.empty() ? 0.0f : Px(ctx, kToastLineGap) + contentSize.y);
        const F32 height = std::ceil(std::max(textHeight, icon) + 2.0f * padding);
        const F32 width = std::ceil(padding + icon + gap + textWidth + padding);
        const Rect card = {std::round(bounds.x + Px(ctx, kToastMargin) - slide), std::round(bottom - height), width,
                           height};
        bottom = card.y - Px(ctx, kToastSpacing);
        item.card = card;

        item.chrome.update({
            .rects = {card},
            .clip = bounds,
            .cornerRadius = radius,
            .softness = softness,
            .drop = Px(ctx, kToastShadowDrop),
            .borderWidth = Px(ctx, 1.0f),
            .lineWidth = Px(ctx, 1.0f),
            .fillKey = "notification_bg",
            .opacity = opacity,
        });
        layoutChild(ctx, item.chrome, bounds);

        const F32 textX = card.x + padding + icon + gap;
        const F32 textY = card.y + std::floor((card.height - textHeight) * 0.5f);
        layoutChild(ctx, item.icon,
                    {card.x + padding + (icon - iconSize.x) * 0.5f, card.y + (card.height - iconSize.y) * 0.5f,
                     iconSize.x, iconSize.y});
        layoutChild(ctx, item.heading, {textX, textY, headingWidth, titleSize});
        if (!item.message.empty()) {
            layoutChild(ctx, item.content, {textX, textY + titleSize + Px(ctx, kToastLineGap), textWidth,
                                            contentSize.y});
        }

        const F32 closeOpacity = item.hovered ? opacity : 0.0f;
        item.close = {std::round(card.x - badge * 0.35f), std::round(card.y - badge * 0.35f), badge, badge};
        auto badgeFill = ctx.color("notification_bg");
        badgeFill = WithAlpha(badgeFill, closeOpacity);
        auto badgeBorder = ctx.color("tile_border");
        badgeBorder = WithAlpha(badgeBorder, closeOpacity);
        item.closeBadge.update({
            .instances = {{.rect = item.close, .visible = closeOpacity > 0.0f, .backgroundColor = badgeFill}},
            .cornerRadius = badge * 0.5f,
            .borderWidth = line,
            .borderColor = badgeBorder,
            .strokeFringe = true,
        });
        layoutChild(ctx, item.closeBadge, bounds);
        item.closeGlyph.update({.str = ICON_LC_X,
                                .colorKey = item.closeHovered ? "tile_button_hover" : "tile_header_text",
                                .scale = kToastCloseGlyph / Typography::FontSize,
                                .opacity = closeOpacity});
        const Extent2D<F32> glyph = measureChild(item.closeGlyph, ctx, {infinite, infinite});
        layoutChild(ctx, item.closeGlyph,
                    {item.close.x + (badge - glyph.x) * 0.5f, item.close.y + (badge - glyph.y) * 0.5f, glyph.x,
                     glyph.y});
        if (!item.hovered) {
            item.close = {};
        }
    }
}

bool ToastLayer::mouseEvent(const MouseEvent& event) {
    const Extent2D<F32> point = event.position;
    switch (event.type) {
        case MouseEventType::Move: {
            auto* target = impl->at(point);
            bool changed = false;
            for (auto& item : impl->items) {
                const bool hovered = item.get() == target;
                const bool closeHovered = hovered && item->close.contains(point.x, point.y);
                if (item->hovered != hovered || item->closeHovered != closeHovered) {
                    item->hovered = hovered;
                    item->closeHovered = closeHovered;
                    changed = true;
                }
            }
            if (changed) {
                invalidate(Dirty::Layout);
            }
            return target != nullptr;
        }
        case MouseEventType::Click: {
            impl->pressedClose = nullptr;
            for (auto& item : impl->items) {
                if (item->live() && item->close.contains(point.x, point.y)) {
                    impl->pressedClose = item.get();
                    return true;
                }
            }
            return impl->at(point) != nullptr;
        }
        case MouseEventType::Release: {
            Impl::Item* pressed = std::exchange(impl->pressedClose, nullptr);
            if (pressed) {
                if (pressed->close.contains(point.x, point.y)) {
                    pressed->closing = true;
                    pressed->hovered = false;
                    pressed->closeHovered = false;
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            return impl->at(point) != nullptr;
        }
        case MouseEventType::Leave:
        case MouseEventType::Cancel: {
            bool changed = false;
            for (auto& item : impl->items) {
                changed = changed || item->hovered || item->closeHovered;
                item->hovered = false;
                item->closeHovered = false;
            }
            impl->pressedClose = nullptr;
            if (changed) {
                invalidate(Dirty::Layout);
            }
            return false;
        }
        default:
            return impl->at(point) != nullptr;
    }
}

std::optional<SurfaceCursor> ToastLayer::cursorAt(const Extent2D<F32>& point) const {
    for (const auto& item : impl->items) {
        if (item->live() && item->close.contains(point.x, point.y)) {
            return SurfaceCursor::Hand;
        }
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
