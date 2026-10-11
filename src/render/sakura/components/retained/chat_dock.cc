#include <jetstream/render/sakura/components/retained/chat_dock.hh>
#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/button.hh>
#include <jetstream/render/sakura/components/retained/label.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/imgui_icons_ext.hh>
#include "../../context.hh"
#include "../../helpers.hh"
#include "../../retained/helpers.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

using Clock = std::chrono::steady_clock;

constexpr F32 kSpeed = 14.0f;
constexpr F32 kHeaderReveal = 0.6f;
constexpr F32 kHeaderInset = 16.0f;
constexpr F32 kButtonSize = 24.0f;
constexpr F32 kButtonGap = 2.0f;
constexpr F32 kHeaderBalance = 10.0f;
constexpr F32 kTitleScale = 1.1f;

enum class Grip { None, Left, Top, Corner };

F32 Settle(F32 value, F32 goal, F32 dt) {
    value = Approach(value, goal, dt * kSpeed);
    return std::abs(goal - value) < 0.001f ? goal : value;
}

F32 Ease(F32 t) {
    return t * t * (3.0f - 2.0f * t);
}

F32 Reveal(F32 expand) {
    return std::clamp((expand - kHeaderReveal) / (1.0f - kHeaderReveal), 0.0f, 1.0f);
}

Chat::ComposerStyle ComposerFor(const ChatDock::Config& config, Chat::ComposerStyle style, bool unread) {
    style.compact = !config.open;
    if (config.open) {
        style.icon.clear();
        style.cornerRadius = config.cornerRadius;
        return style;
    }
    style.borderColorKey = unread ? config.alertColorKey : config.outlineColorKey;
    style.borderWidth = config.outlineWidth;
    return style;
}

}  // namespace

struct ChatDock::Impl {
    explicit Impl(ChatDock& self) : self(self) {}

    struct Header {
        Rect rect;
        F32 alpha = 0.0f;
        bool visible = false;
    };

    ChatDock& self;
    Config config;
    Chat::Config chatConfig;
    bool chatChanged = true;
    F32 transcriptOpacity = -1.0f;
    Box background;
    Chat chat;
    Label title;
    Button clear;
    Button minimize;

    std::function<void()> onClear;
    bool conversation = false;
    bool streaming = false;
    bool unread = false;
    F32 slide = 0.0f;
    F32 expand = 0.0f;
    FrameClock clock;

    bool sized = false;
    Extent2D<F32> size;
    Rect card;
    Grip hover = Grip::None;
    Grip drag = Grip::None;
    Extent2D<F32> grab;
    Extent2D<F32> pointer;

    F32 unit() const {
        return config.fontSize / Typography::FontSize;
    }

    Extent2D<F32> clampSize(Extent2D<F32> value, const Rect& bounds) const {
        const F32 margin = 2.0f * config.margin;
        const auto limit = [](F32 v, F32 lo, F32 hi) {
            return std::max(lo, std::min(v, std::max(lo, hi)));
        };
        return {limit(value.x, config.minSize.x, bounds.width - margin),
                limit(value.y, config.minSize.y, bounds.height - margin)};
    }

    Grip gripAt(F32 x, F32 y) const {
        if (!config.open || expand < 1.0f || !card.contains(x, y)) {
            return Grip::None;
        }
        const F32 grip = config.resizeGrip;
        const bool left = x < card.x + grip;
        const bool top = y < card.y + grip;
        return left && top ? Grip::Corner : left ? Grip::Left : top ? Grip::Top : Grip::None;
    }

    void animate(const Context& ctx) {
        const F32 dt = clock.tick(Clock::now(), 0.1f);
        const bool peek = config.open || ctx.windowFocused;
        slide = Settle(slide, peek ? 1.0f : 0.0f, dt);
        expand = Settle(expand, config.open ? 1.0f : 0.0f, dt);
    }

    void applyDrag(const Rect& bounds) {
        if (drag == Grip::None) {
            return;
        }
        auto dragged = size;
        if (drag != Grip::Top) {
            dragged.x = bounds.right() - config.margin - (pointer.x - grab.x);
        }
        if (drag != Grip::Left) {
            dragged.y = bounds.bottom() - config.margin - (pointer.y - grab.y);
        }
        size = clampSize(dragged, bounds);
    }

    Rect cardRect(const Context& ctx, const Rect& bounds) {
        const auto clamped = clampSize(size, bounds);
        const F32 margin = config.margin;
        const F32 fullWidth = std::clamp(clamped.x, 0.0f, std::max(0.0f, bounds.width - 2.0f * margin));
        const F32 width = Lerp(std::min(fullWidth, config.compactWidth), fullWidth, Ease(slide));
        const F32 barHeight = self.measureChild(chat, ctx, {width, std::numeric_limits<F32>::infinity()}).y;
        const F32 openHeight = std::max(barHeight, std::min(clamped.y, bounds.height - 2.0f * margin));
        const F32 height = Lerp(barHeight, openHeight, Ease(expand));
        return {bounds.right() - margin - width, bounds.bottom() - margin - height, width, height};
    }

    void applyCursor() const {
        switch (drag != Grip::None ? drag : hover) {
            case Grip::Left:
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                break;
            case Grip::Top:
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                break;
            case Grip::Corner:
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
                break;
            case Grip::None:
                break;
        }
    }

    Header headerFor() const {
        const F32 alpha = Reveal(expand);
        return {
            .rect = {card.x, card.y, card.width, config.headerHeight + kHeaderBalance * unit()},
            .alpha = alpha,
            .visible = alpha > 0.0f && !card.empty(),
        };
    }

    void layoutBackground(const Context& ctx, const Rect& bounds) {
        const auto panel = ctx.color(config.backgroundColorKey);
        const auto tint = ctx.color(config.outlineColorKey);
        const ColorRGBA<F32> fill = {panel.r, panel.g, panel.b, panel.a * expand};
        const auto outline = Over(panel, tint, fill.a);
        background.update({
            .id = config.id + ":bg",
            .instances = {{.rect = card, .visible = expand > 0.0f && !card.empty(), .backgroundColor = fill}},
            .cornerRadius = config.cardRadius,
            .borderWidth = config.outlineWidth,
            .borderColor = outline,
        });
        self.layoutChild(ctx, background, bounds);
    }

    void layoutTitle(const Context& ctx, const Rect& bounds, const Header& header) {
        const F32 inset = kHeaderInset * unit();
        auto titleColor = ctx.color("text_secondary");
        titleColor.a *= header.alpha;
        title.update({
            .id = config.id + ":title",
            .instances = {{
                .rect = {header.rect.x + inset, header.rect.y,
                         std::max(0.0f, header.rect.width - 2.0f * inset), header.rect.height},
                .str = config.title,
                .visible = header.visible,
                .color = titleColor,
                .fontSize = config.fontSize * kTitleScale,
                .alignment = {0, 1},
            }},
            .clip = card,
            .fontName = "default_body_bold",
        });
        self.layoutChild(ctx, title, bounds);
    }

    Button::Config iconButton(const std::string& id, const char* icon, std::function<void()> onClick) const {
        return {
            .id = config.id + id,
            .str = icon,
            .colorKey = "transparent",
            .hoveredColorKey = "button_hovered",
            .activeColorKey = "button_active",
            .borderColorKey = "transparent",
            .textColorKey = "text_secondary",
            .fontSize = config.fontSize * 0.85f,
            .fontName = Typography::IconFont,
            .cornerRadius = kButtonSize * unit() * 0.5f,
            .onClick = std::move(onClick),
        };
    }

    void layoutButtons(const Context& ctx, const Header& header) {
        const F32 inset = kHeaderInset * unit();
        const F32 buttonSize = kButtonSize * unit();
        const Rect minimizeRect = header.visible
            ? Rect{header.rect.right() - inset * 0.5f - buttonSize,
                   header.rect.y + (header.rect.height - buttonSize) * 0.5f,
                   buttonSize, buttonSize}
            : Rect{};
        minimize.update(iconButton(":minimize", ICON_FA_MINUS, [this]() {
            if (config.onMinimize) {
                config.onMinimize();
            }
        }));
        self.layoutChild(ctx, minimize, minimizeRect);

        const Rect clearRect = header.visible
            ? Rect{minimizeRect.x - kButtonGap * unit() - buttonSize, minimizeRect.y, buttonSize, buttonSize}
            : Rect{};
        clear.update(iconButton(":clear", ICON_FA_PEN_TO_SQUARE, [this]() {
            if (onClear) {
                onClear();
            }
        }));
        self.layoutChild(ctx, clear, clearRect);
    }
};

ChatDock::ChatDock() {
    impl = std::make_unique<Impl>(*this);
    add(impl->background);
    add(impl->chat);
    add(impl->title);
    add(impl->clear);
    add(impl->minimize);
}

ChatDock::~ChatDock() = default;

bool ChatDock::update(Config config) {
    if (!impl->sized) {
        impl->size = config.size;
        impl->sized = true;
    } else if (impl->config.fontSize > 0.0f && config.fontSize != impl->config.fontSize) {
        const F32 scale = config.fontSize / impl->config.fontSize;
        impl->size = {impl->size.x * scale, impl->size.y * scale};
    }
    if (config.open != impl->config.open || config.title != impl->config.title) {
        invalidate(Dirty::Paint);
    }
    impl->conversation = !config.chat.snapshot.messages.empty();
    const bool streaming = config.chat.snapshot.streaming;
    if (impl->streaming && !streaming && !config.open) {
        impl->unread = true;
    }
    if (config.open || !impl->conversation) {
        impl->unread = false;
    }
    impl->streaming = streaming;
    impl->onClear = config.chat.onClear;
    config.chat.composer = ComposerFor(config, std::move(config.chat.composer), impl->unread);
    impl->chatConfig = std::move(config.chat);
    impl->chatChanged = true;
    impl->config = std::move(config);
    return true;
}

bool ChatDock::event(const MouseEvent& event) {
    auto& state = *impl;
    const auto& point = event.position;
    switch (event.type) {
        case MouseEventType::Click:
            if (event.button == MouseButton::Left && !state.config.open && state.conversation &&
                state.card.contains(point.x, point.y) && state.config.onOpen) {
                state.config.onOpen();
            }
            if (event.button == MouseButton::Left) {
                state.drag = state.gripAt(point.x, point.y);
                if (state.drag != Grip::None) {
                    state.grab = {point.x - state.card.x, point.y - state.card.y};
                    state.pointer = point;
                    return true;
                }
            }
            break;
        case MouseEventType::Move:
            if (state.drag != Grip::None) {
                state.pointer = point;
                invalidate(Dirty::Paint);
                return true;
            }
            state.hover = state.gripAt(point.x, point.y);
            break;
        case MouseEventType::Release:
            if (state.drag != Grip::None) {
                state.drag = Grip::None;
                return true;
            }
            break;
        case MouseEventType::Leave:
            state.hover = Grip::None;
            break;
        default:
            break;
    }
    return Component::event(event);
}

bool ChatDock::hitTest(const Extent2D<F32>& point) const {
    return impl->drag != Grip::None || impl->card.contains(point.x, point.y);
}

void ChatDock::layout(const Context& ctx) {
    auto& state = *impl;
    const Rect bounds = frame();

    state.animate(ctx);
    const F32 opacity = Reveal(state.expand);
    if (state.chatChanged || opacity != state.transcriptOpacity) {
        auto chatConfig = state.chatConfig;
        chatConfig.transcriptOpacity = opacity;
        state.chat.update(std::move(chatConfig));
        state.chatChanged = false;
        state.transcriptOpacity = opacity;
    }
    state.applyDrag(bounds);
    state.card = state.cardRect(ctx, bounds);
    state.applyCursor();

    const auto header = state.headerFor();
    const F32 headerHeight = state.config.headerHeight * Ease(state.expand);
    const Rect& card = state.card;
    layoutChild(ctx, state.chat, {card.x, card.y + headerHeight, card.width, std::max(0.0f, card.height - headerHeight)});
    state.layoutBackground(ctx, bounds);
    state.layoutTitle(ctx, bounds, header);
    state.layoutButtons(ctx, header);
}

}  // namespace Jetstream::Sakura::Retained
