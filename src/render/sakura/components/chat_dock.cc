#include <jetstream/render/sakura/components/chat_dock.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kSpeed = 14.0f;
constexpr F32 kMaxStep = 0.1f;
constexpr F32 kSettled = 0.001f;
constexpr F32 kHeaderReveal = 0.6f;
constexpr F32 kHeaderInset = 16.0f;
constexpr F32 kHeaderBalance = 10.0f;
constexpr F32 kButtonSize = 24.0f;
constexpr F32 kButtonGap = 2.0f;
constexpr F32 kIconScale = 0.85f;
constexpr F32 kTitleScale = 1.1f;
constexpr U64 kTitleCharacters = 64;

enum class Grip { None, Left, Top, Corner };

F32 Mix(F32 from, F32 to, F32 t) {
    return from * (1.0f - t) + to * t;
}

F32 Settle(F32 value, F32 goal, F32 dt) {
    value = Mix(value, goal, 1.0f - std::exp(-dt * kSpeed));
    return std::abs(goal - value) < kSettled ? goal : value;
}

F32 Ease(F32 t) {
    return t * t * (3.0f - 2.0f * t);
}

F32 Reveal(F32 expand) {
    return std::clamp((expand - kHeaderReveal) / (1.0f - kHeaderReveal), 0.0f, 1.0f);
}

SurfaceCursor CursorFor(Grip grip) {
    switch (grip) {
        case Grip::Left:
            return SurfaceCursor::ResizeEW;
        case Grip::Top:
            return SurfaceCursor::ResizeNS;
        case Grip::Corner:
        case Grip::None:
            break;
    }
    return SurfaceCursor::ResizeNWSE;
}

}  // namespace

struct ChatDock::Impl {
    Config config;
    Box background;
    Chat chat;
    Label title;
    Button clear;
    Button minimize;

    bool sized = false;
    Extent2D<F32> size = {0.0f, 0.0f};
    bool conversation = false;
    bool streaming = false;
    bool unread = false;
    bool inputFocused = false;
    F32 slide = 0.0f;
    F32 expand = 0.0f;
    F32 opacity = 0.0f;

    Rect bounds;
    Rect card;
    F32 ratio = 1.0f;
    F32 margin = 0.0f;
    F32 grip = 0.0f;
    Extent2D<F32> minimum = {0.0f, 0.0f};
    Grip drag = Grip::None;
    Extent2D<F32> grab = {0.0f, 0.0f};

    F32 unit() const {
        return config.fontSize / Typography::FontSize;
    }

    Chat::ComposerStyle composerStyle() const {
        auto style = config.chat.composer;
        style.compact = !config.open;
        if (config.open) {
            style.icon.clear();
            style.rounding = config.rounding;
            return style;
        }
        style.borderColorKey = unread ? config.alertColorKey : config.outlineColorKey;
        style.borderWidth = config.outlineWidth;
        return style;
    }

    void pushChat(ChatDock& self) {
        auto next = config.chat;
        next.composer = composerStyle();
        next.transcriptOpacity = opacity;
        next.onFocusChange = [this, &self](bool focused) {
            if (focused != inputFocused) {
                inputFocused = focused;
                self.invalidate(Dirty::Layout);
            }
            if (config.chat.onFocusChange) {
                config.chat.onFocusChange(focused);
            }
        };
        chat.update(std::move(next));
    }

    Button::Config iconButton(const char* icon, Callback<void()> onClick) const {
        const F32 side = kButtonSize * unit();
        return {
            .str = icon,
            .size = {side, side},
            .colorKey = "transparent",
            .hoveredColorKey = "button_hovered",
            .activeColorKey = "button_active",
            .borderColorKey = "transparent",
            .textColorKey = "text_secondary",
            .textScale = kIconScale * unit(),
            .rounding = side * 0.5f,
            .padding = {0.0f, 0.0f},
            .onClick = std::move(onClick),
        };
    }

    Extent2D<F32> clampSize(Extent2D<F32> value) const {
        const auto limit = [](F32 length, F32 low, F32 high) {
            return std::max(low, std::min(length, std::max(low, high)));
        };
        return {limit(value.x, minimum.x, bounds.width - 2.0f * margin),
                limit(value.y, minimum.y, bounds.height - 2.0f * margin)};
    }

    Rect cardRect(ChatDock& self, const Context& ctx) {
        const auto clamped = clampSize({Px(ctx, size.x), Px(ctx, size.y)});
        const F32 fullWidth = std::clamp(clamped.x, 0.0f, std::max(0.0f, bounds.width - 2.0f * margin));
        const F32 width = Mix(std::min(fullWidth, Px(ctx, config.compactWidth)), fullWidth, Ease(slide));
        const F32 barHeight = self.measureChild(chat, ctx, {width, std::numeric_limits<F32>::infinity()}).y;
        const F32 openHeight = std::max(barHeight, std::min(clamped.y, bounds.height - 2.0f * margin));
        const F32 height = Mix(barHeight, openHeight, Ease(expand));
        return {bounds.right() - margin - width, bounds.bottom() - margin - height, width, height};
    }

    Grip gripAt(const Extent2D<F32>& point) const {
        if (!config.open || expand < 1.0f || !card.contains(point.x, point.y)) {
            return Grip::None;
        }
        const bool left = point.x < card.x + grip;
        const bool top = point.y < card.y + grip;
        return left && top ? Grip::Corner : left ? Grip::Left : top ? Grip::Top : Grip::None;
    }

    bool resize(const Extent2D<F32>& point) {
        const F32 scale = std::max(ratio, 1e-3f);
        Extent2D<F32> dragged = {size.x * scale, size.y * scale};
        if (drag != Grip::Top) {
            dragged.x = bounds.right() - margin - (point.x - grab.x);
        }
        if (drag != Grip::Left) {
            dragged.y = bounds.bottom() - margin - (point.y - grab.y);
        }
        const auto clamped = clampSize(dragged);
        const Extent2D<F32> next = {
            drag != Grip::Top ? clamped.x / scale : size.x,
            drag != Grip::Left ? clamped.y / scale : size.y,
        };
        if (next == size) {
            return false;
        }
        size = next;
        return true;
    }

    void reveal(ChatDock& self, Component& child, bool shown) {
        if (child.visible() == shown) {
            return;
        }
        if (!shown) {
            self.cancelMouse(child);
        }
        child.setVisible(shown);
    }
};

ChatDock::ChatDock() {
    impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->chat);
    addChild(impl->title);
    addChild(impl->clear);
    addChild(impl->minimize);
    impl->clear.setVisible(false);
    impl->minimize.setVisible(false);
}

ChatDock::~ChatDock() = default;

void ChatDock::update(Config config) {
    auto& state = *impl;
    if (!state.sized || config.size != state.config.size) {
        state.size = config.size;
        state.sized = true;
    }
    const bool conversation = !config.chat.snapshot.messages.empty();
    const bool streaming = config.chat.snapshot.streaming;
    bool unread = state.unread || (state.streaming && !streaming && !config.open);
    if (config.open || !conversation) {
        unread = false;
    }
    if (config != state.config || unread != state.unread) {
        invalidate(Dirty::Layout);
    }
    if (!config.open) {
        state.drag = Grip::None;
    }
    state.conversation = conversation;
    state.streaming = streaming;
    state.unread = unread;
    state.config = std::move(config);
    state.pushChat(*this);
    state.clear.update(state.iconButton(ICON_LC_SQUARE_PEN, [this]() {
        if (impl->config.chat.onClear) {
            impl->config.chat.onClear();
        }
    }));
    state.minimize.update(state.iconButton(ICON_LC_MINUS, [this]() {
        if (!requestFocus()) {
            return;
        }
        releaseFocus();
        if (impl->config.onMinimize) {
            impl->config.onMinimize();
        }
    }));
}

bool ChatDock::covers(const Extent2D<F32>& point) const {
    return impl->drag != Grip::None || impl->card.contains(point.x, point.y);
}

void ChatDock::layout(const Context& ctx) {
    auto& state = *impl;
    const auto& config = state.config;
    const Rect area = frame();
    const F32 top = std::clamp(Px(ctx, config.topOffset), 0.0f, area.height);
    state.bounds = {area.x, area.y + top, area.width, area.height - top};
    state.ratio = ctx.pixelRatio;
    state.margin = Px(ctx, config.margin);
    state.grip = Px(ctx, config.resizeGrip);
    state.minimum = {Px(ctx, config.minimumSize.x), Px(ctx, config.minimumSize.y)};
    const Rect& bounds = state.bounds;

    const F32 dt = std::clamp(ctx.deltaTime, 0.0f, kMaxStep);
    const F32 slideGoal = config.open || state.inputFocused ? 1.0f : 0.0f;
    const F32 expandGoal = config.open ? 1.0f : 0.0f;
    state.slide = Settle(state.slide, slideGoal, dt);
    state.expand = Settle(state.expand, expandGoal, dt);
    if (state.slide != slideGoal || state.expand != expandGoal) {
        requestFrame();
    }

    const F32 reveal = Reveal(state.expand);
    if (reveal != state.opacity) {
        state.opacity = reveal;
        state.pushChat(*this);
    }

    Context inner = ctx;
    if (state.expand > 0.0f) {
        inner.backdropColorKey = config.backgroundColorKey;
        inner.backdropTopColorKey.clear();
    }
    state.card = state.cardRect(*this, inner);
    const Rect card = state.card;

    const auto panel = ctx.color(config.backgroundColorKey);
    const auto tint = ctx.color(config.outlineColorKey);
    const F32 alpha = panel.a * std::min(1.0f, state.expand / kHeaderReveal);
    state.background.update({
        .instances = {{
            .rect = card,
            .visible = state.expand > 0.0f && !card.empty(),
            .backgroundColor = {panel.r, panel.g, panel.b, alpha},
        }},
        .cornerRadius = Px(ctx, config.cardRounding),
        .borderWidth = Px(ctx, config.outlineWidth),
        .borderColor = {panel.r + (tint.r - panel.r) * tint.a, panel.g + (tint.g - panel.g) * tint.a,
                        panel.b + (tint.b - panel.b) * tint.a, alpha},
    });
    layoutChild(ctx, state.background, bounds);

    const F32 headerOffset = Px(ctx, config.headerHeight) * Ease(state.expand);
    layoutChild(inner, state.chat,
                {card.x, card.y + headerOffset, card.width, std::max(0.0f, card.height - headerOffset)});

    const F32 unit = state.unit();
    const bool headerShown = reveal > 0.0f && !card.empty();
    const Rect header = {card.x, card.y, card.width, Px(ctx, config.headerHeight + kHeaderBalance * unit)};
    const F32 inset = Px(ctx, kHeaderInset * unit);
    state.title.update({
        .instances = {{
            .rect = {header.x + inset, header.y, std::max(0.0f, header.width - 2.0f * inset), header.height},
            .str = config.title,
            .visible = headerShown,
            .color = WithAlpha(ctx.color("text_secondary"), reveal),
            .fontSize = Px(ctx, config.fontSize * kTitleScale),
            .alignment = {0, 1},
        }},
        .clip = card,
        .fontName = Typography::BodyBoldFont,
        .maxCharacters = kTitleCharacters,
    });
    layoutChild(ctx, state.title, bounds);

    const F32 button = Px(ctx, kButtonSize * unit);
    const Rect minimizeRect = {header.right() - inset * 0.5f - button, header.y + (header.height - button) * 0.5f,
                               button, button};
    const Rect clearRect = {minimizeRect.x - Px(ctx, kButtonGap * unit) - button, minimizeRect.y, button, button};
    state.reveal(*this, state.minimize, headerShown);
    state.reveal(*this, state.clear, headerShown);
    layoutChild(ctx, state.minimize, minimizeRect);
    layoutChild(ctx, state.clear, clearRect);
}

bool ChatDock::mouseEvent(const MouseEvent& event) {
    auto& state = *impl;
    const Extent2D<F32> point = event.position;
    switch (event.type) {
        case MouseEventType::Click:
            if (event.button != MouseButton::Left) {
                break;
            }
            if (!state.config.open && state.conversation && state.card.contains(point.x, point.y) &&
                state.config.onOpen) {
                state.config.onOpen();
            }
            state.drag = state.gripAt(point);
            if (state.drag != Grip::None) {
                state.grab = {point.x - state.card.x, point.y - state.card.y};
                requestCursor(CursorFor(state.drag));
                return true;
            }
            break;
        case MouseEventType::Move:
            if (state.drag != Grip::None) {
                requestCursor(CursorFor(state.drag));
                if (state.resize(point)) {
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            break;
        case MouseEventType::Release:
            if (state.drag != Grip::None && event.button == MouseButton::Left) {
                state.drag = Grip::None;
                return true;
            }
            break;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            state.drag = Grip::None;
            break;
        default:
            break;
    }
    return mouseEventChildren(event);
}

std::optional<SurfaceCursor> ChatDock::cursorOverChildren(const Extent2D<F32>& point) const {
    const Grip grip = impl->drag != Grip::None ? impl->drag : impl->gripAt(point);
    if (grip == Grip::None) {
        return std::nullopt;
    }
    return CursorFor(grip);
}

}  // namespace Jetstream::Sakura
