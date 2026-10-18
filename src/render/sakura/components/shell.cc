#include <jetstream/render/sakura/components/shell.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/controls.hh>
#include <jetstream/render/sakura/components/layout.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/imgui_icons_ext.hh>

#include "../context.hh"
#include "../helpers.hh"
#include "../metrics.hh"
#include "../toast_queue.hh"
#include "../detail/card_set.hh"
#include "../detail/helpers.hh"
#include "../detail/access.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kFloatingTitleHeight = 32.0f;
constexpr F32 kFloatingTitleSize = 16.0f;
constexpr F32 kFloatingTitlePadX = 12.0f;
constexpr F32 kFloatingCloseButton = 20.0f;
constexpr F32 kFloatingCloseGlyph = 13.0f;
constexpr F32 kFloatingCascade = 24.0f;
constexpr U64 kFloatingCascadeSteps = 8;
constexpr F32 kResizeGrip = 16.0f;
constexpr F32 kResizeSlack = 4.0f;
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

Rect Anchored(const Rect& bounds, Anchor anchor, Extent2D<F32> size, Extent2D<F32> offset, F32 margin) {
    switch (anchor) {
        case Anchor::Fill:
            return bounds;
        case Anchor::TopLeft:
            return {bounds.x + margin + offset.x, bounds.y + margin + offset.y, size.x, size.y};
        case Anchor::TopCenter:
            return {bounds.x + std::floor((bounds.width - size.x) * 0.5f) + offset.x, bounds.y + margin + offset.y,
                    size.x, size.y};
        case Anchor::TopRight:
            return {bounds.right() - margin - size.x + offset.x, bounds.y + margin + offset.y, size.x, size.y};
        case Anchor::Center:
            return {bounds.x + std::floor((bounds.width - size.x) * 0.5f) + offset.x,
                    bounds.y + std::floor((bounds.height - size.y) * 0.5f) + offset.y, size.x, size.y};
        case Anchor::BottomLeft:
            return {bounds.x + margin + offset.x, bounds.bottom() - margin - size.y + offset.y, size.x, size.y};
        case Anchor::BottomCenter:
            return {bounds.x + std::floor((bounds.width - size.x) * 0.5f) + offset.x,
                    bounds.bottom() - margin - size.y + offset.y, size.x, size.y};
        case Anchor::BottomRight:
            return {bounds.right() - margin - size.x + offset.x, bounds.bottom() - margin - size.y + offset.y,
                    size.x, size.y};
    }
    return bounds;
}

}  // namespace

struct Overlay::Impl {
    Config config;
};

Overlay::Overlay() {
    impl = std::make_unique<Impl>();
}

Overlay::~Overlay() = default;

void Overlay::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> children;
    for (const auto& item : config.items) {
        if (item.child) {
            children.push_back(item.child);
        }
    }
    mountChildren(children);
    impl->config = std::move(config);
}

Extent2D<F32> Overlay::measure(const Context&, Extent2D<F32> available) {
    return {FillLength(available.x), FillLength(available.y)};
}

void Overlay::layout(const Context& ctx) {
    const Rect bounds = frame();
    for (const auto& item : impl->config.items) {
        if (!item.child || !item.child->visible()) {
            continue;
        }
        Extent2D<F32> size = {ResolveLength(ctx, item.size.x, 0.0f, bounds.width),
                              ResolveLength(ctx, item.size.y, 0.0f, bounds.height)};
        if (item.anchor != Anchor::Fill && (item.size.x == 0.0f || item.size.y == 0.0f)) {
            const auto measured = measureChild(*item.child, ctx, {item.size.x != 0.0f ? size.x : bounds.width,
                                                                  item.size.y != 0.0f ? size.y : bounds.height});
            size.x = item.size.x != 0.0f ? size.x : std::min(measured.x, bounds.width);
            size.y = item.size.y != 0.0f ? size.y : std::min(measured.y, bounds.height);
        }
        const Rect rect = Anchored(bounds, item.anchor, size, {Px(ctx, item.offset.x), Px(ctx, item.offset.y)}, 0.0f);
        layoutChild(ctx, *item.child, rect);
    }
}

struct HudLayer::Impl {
    Config config;
    std::vector<std::unique_ptr<Box>> frames;
    std::vector<Rect> rects;
};

HudLayer::HudLayer() {
    impl = std::make_unique<Impl>();
}

HudLayer::~HudLayer() = default;

void HudLayer::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    while (impl->frames.size() < impl->config.huds.size()) {
        impl->frames.push_back(std::make_unique<Box>());
    }
    std::vector<Component*> children;
    for (U64 index = 0; index < impl->config.huds.size(); ++index) {
        children.push_back(impl->frames[index].get());
        if (impl->config.huds[index].content) {
            children.push_back(impl->config.huds[index].content);
        }
    }
    mountChildren(children);
}

bool HudLayer::covers(const Extent2D<F32>& point) const {
    for (const auto& rect : impl->rects) {
        if (rect.contains(point.x, point.y)) {
            return true;
        }
    }
    return false;
}

void HudLayer::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    impl->rects.clear();
    for (U64 index = 0; index < impl->config.huds.size(); ++index) {
        const auto& hud = impl->config.huds[index];
        const Extent2D<F32> padding = {Px(ctx, hud.padding.x), Px(ctx, hud.padding.y)};
        const F32 margin = Px(ctx, hud.margin);
        const Extent2D<F32> room = {std::max(0.0f, bounds.width - 2.0f * margin),
                                    std::max(0.0f, bounds.height - 2.0f * margin)};
        Extent2D<F32> size = {ResolveLength(ctx, hud.size.x, 0.0f, room.x), ResolveLength(ctx, hud.size.y, 0.0f, room.y)};
        if (hud.content && (hud.size.x == 0.0f || hud.size.y == 0.0f)) {
            const F32 width = hud.size.x != 0.0f ? size.x - 2.0f * padding.x : std::max(0.0f, room.x - 2.0f * padding.x);
            const auto measured = measureChild(*hud.content, ctx, {width, infinite});
            if (hud.size.x == 0.0f) {
                size.x = std::ceil(measured.x + 2.0f * padding.x);
            }
            if (hud.size.y == 0.0f) {
                size.y = std::ceil(measured.y + 2.0f * padding.y);
            }
        }
        const Rect rect = Anchored(bounds, hud.anchor, size, {Px(ctx, hud.offset.x), Px(ctx, hud.offset.y)}, margin);
        impl->rects.push_back(rect);
        impl->frames[index]->update({
            .instances = {{.rect = rect, .visible = !rect.empty(), .backgroundColor = ctx.color(hud.backgroundColorKey)}},
            .cornerRadius = Px(ctx, hud.rounding),
            .borderWidth = hud.borderSize > 0.0f ? Px(ctx, hud.borderSize * 2.0f) : 0.0f,
            .borderColor = ctx.color(hud.borderColorKey),
            .strokeFringe = true,
        });
        layoutChild(ctx, *impl->frames[index], rect);
        if (hud.content) {
            layoutChild(ctx, *hud.content, {rect.x + padding.x, rect.y + padding.y,
                                            std::max(0.0f, rect.width - 2.0f * padding.x),
                                            std::max(0.0f, rect.height - 2.0f * padding.y)});
        }
    }
}

bool HudLayer::mouseEvent(const MouseEvent& event) {
    const bool handled = mouseEventChildren(event);
    if (handled || event.type != MouseEventType::Click || event.button != MouseButton::Left) {
        return handled;
    }
    for (U64 index = 0; index < impl->rects.size() && index < impl->config.huds.size(); ++index) {
        const auto& hud = impl->config.huds[index];
        if (hud.clickable && hud.onClick && impl->rects[index].contains(event.position.x, event.position.y)) {
            hud.onClick();
            return true;
        }
    }
    return false;
}

std::optional<SurfaceCursor> HudLayer::cursorAt(const Extent2D<F32>& point) const {
    for (U64 index = 0; index < impl->rects.size() && index < impl->config.huds.size(); ++index) {
        if (impl->config.huds[index].clickable && impl->rects[index].contains(point.x, point.y)) {
            return SurfaceCursor::Hand;
        }
    }
    return std::nullopt;
}

struct ModalLayer::Impl {
    Config config;
    Box dim;
    Box window;
    Divider divider;
    Button close;
    Stack stack;
    Rect rect;
};

ModalLayer::ModalLayer() {
    impl = std::make_unique<Impl>();
    impl->divider.update({});
    impl->close.update({
        .str = "Close",
        .size = {Fill, 40.0f},
        .onClick = [impl = impl.get()]() {
            if (impl->config.onClose) {
                impl->config.onClose();
            }
        },
    });
}

ModalLayer::~ModalLayer() = default;

void ModalLayer::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (!impl->config.content) {
        mountChildren({});
        Access::prepareInput(*this);
        return;
    }
    const bool fixedHeight = impl->config.size.y != 0.0f;
    impl->stack.update({
        .items = {
            {.child = impl->config.content, .grow = fixedHeight ? 1.0f : 0.0f},
            {.child = &impl->divider},
            {.child = &impl->close},
        },
    });
    mountChildren({&impl->dim, &impl->window, &impl->stack});
    Access::prepareInput(*this);
}

bool ModalLayer::modal() const {
    return impl->config.content != nullptr;
}

bool ModalLayer::covers(const Extent2D<F32>&) const {
    return impl->config.content != nullptr;
}

void ModalLayer::layout(const Context& ctx) {
    if (!impl->config.content) {
        return;
    }
    const Rect bounds = frame();
    const auto& config = impl->config;
    const F32 padding = Px(ctx, kWindowPadding);
    const F32 margin = Px(ctx, 19.0f);
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const Extent2D<F32> room = {std::max(0.0f, bounds.width - 2.0f * margin), std::max(0.0f, bounds.height - 2.0f * margin)};
    F32 natural = 0.0f;
    if (config.size.x == 0.0f) {
        natural = measureChild(impl->stack, ctx, {infinite, infinite}).x + 2.0f * padding;
    }
    const F32 width = ResolveLength(ctx, config.size.x, natural, room.x);
    natural = 0.0f;
    if (config.size.y == 0.0f) {
        natural = measureChild(impl->stack, ctx, {std::max(0.0f, width - 2.0f * padding), infinite}).y + 2.0f * padding;
    }
    const F32 height = ResolveLength(ctx, config.size.y, natural, room.y);
    const Rect rect = {std::round(bounds.x + (bounds.width - width) * 0.5f),
                       std::round(bounds.y + (bounds.height - height) * 0.5f), std::ceil(width), std::ceil(height)};
    impl->rect = rect;

    impl->dim.update({.instances = {{.rect = bounds, .visible = true, .backgroundColor = ctx.color("modal_dim")}}});
    layoutChild(ctx, impl->dim, bounds);
    impl->window.update({
        .instances = {{.rect = rect, .visible = !rect.empty(), .backgroundColor = ctx.color("popup_bg")}},
        .cornerRadius = Px(ctx, kPanelRounding),
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color("border"),
        .strokeFringe = true,
    });
    layoutChild(ctx, impl->window, rect);
    layoutChild(ctx, impl->stack, Inset(rect, padding));
}

bool ModalLayer::mouseEvent(const MouseEvent& event) {
    if (!impl->config.content) {
        return false;
    }
    mouseEventChildren(event);
    return true;
}

bool ModalLayer::shortcutEvent(const KeyEvent& event) {
    if (!impl->config.content || event.type != KeyEventType::Press || event.key != KeyCode::Escape || event.repeat) {
        return false;
    }
    if (impl->config.onClose) {
        impl->config.onClose();
    }
    return true;
}

struct FloatingLayer::Impl {
    struct State {
        Rect rect;
        bool placed = false;
    };

    struct Chrome {
        Box shadow;
        Box card;
        Text label;
        Text close;
        Box border;
        Box highlight;
        Box rim;
    };

    enum class Drag {
        None,
        Move,
        Resize,
    };

    Config config;
    std::unordered_map<std::string, State> states;
    std::vector<std::string> order;
    std::unordered_map<std::string, std::unique_ptr<Chrome>> chrome;
    std::string focused;
    std::string closeHovered;
    std::string pointerWindow;
    std::string capturedWindow;
    std::array<bool, 3> capturedButtons{};
    Drag drag = Drag::None;
    std::string dragTarget;
    Extent2D<F32> dragStart = {0.0f, 0.0f};
    Rect dragRect;
    F32 titleHeight = 0.0f;
    F32 titlePadX = 0.0f;
    F32 closeButton = 0.0f;
    F32 closeGlyph = 0.0f;
    F32 grip = 0.0f;
    F32 slack = 0.0f;
    U64 cascade = 0;

    const Window* window(const std::string& id) const {
        for (const auto& item : config.windows) {
            if (item.id == id) {
                return &item;
            }
        }
        return nullptr;
    }

    Text::Config titleConfig(const Window& item) const {
        return {
            .str = item.title,
            .font = Text::Font::DisplaySemiBold,
            .colorKey = focused == item.id ? "text_primary" : "tile_header_text",
            .scale = kFloatingTitleSize / Typography::FontSize,
        };
    }

    Text::Config closeConfig(const std::string& id) const {
        return {
            .str = ICON_FA_XMARK,
            .colorKey = closeHovered == id ? "tile_button_hover" : "tile_header_text",
            .scale = kFloatingCloseGlyph / Typography::FontSize,
        };
    }

    Rect titleRect(const Rect& rect) const {
        return {rect.x, rect.y, rect.width, titleHeight};
    }

    Rect bodyRect(const Rect& rect) const {
        return {rect.x, rect.y + titleHeight, rect.width, std::max(0.0f, rect.height - titleHeight)};
    }

    Rect closeRect(const Rect& rect) const {
        const F32 centerX = rect.right() - titlePadX - closeGlyph * 0.5f;
        const F32 centerY = rect.y + titleHeight * 0.5f;
        return {std::round(centerX - closeButton * 0.5f), std::round(centerY - closeButton * 0.5f), closeButton,
                closeButton};
    }

    Rect gripRect(const Rect& rect) const {
        return {rect.right() - grip, rect.bottom() - grip, grip + slack, grip + slack};
    }

    std::string windowAt(const Extent2D<F32>& point) const {
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const auto state = states.find(*it);
            if (state == states.end()) {
                continue;
            }
            const Rect& rect = state->second.rect;
            if (rect.contains(point.x, point.y) || gripRect(rect).contains(point.x, point.y)) {
                return *it;
            }
        }
        return {};
    }

    std::vector<Component*> parts() const {
        std::vector<Component*> result;
        for (const auto& id : order) {
            const auto* item = window(id);
            const auto found = chrome.find(id);
            if (!item || found == chrome.end()) {
                continue;
            }
            auto& parts = *found->second;
            result.insert(result.end(), {&parts.shadow, &parts.card, &parts.label, &parts.close});
            if (item->content) {
                result.push_back(item->content);
            }
            result.insert(result.end(), {&parts.border, &parts.highlight, &parts.rim});
        }
        return result;
    }

    void raise(const std::string& id) {
        const auto it = std::find(order.begin(), order.end(), id);
        if (it != order.end() && it + 1 != order.end()) {
            order.erase(it);
            order.push_back(id);
        }
    }
};

FloatingLayer::FloatingLayer() {
    impl = std::make_unique<Impl>();
}

FloatingLayer::~FloatingLayer() = default;

void FloatingLayer::update(Config config) {
    if (!impl->capturedWindow.empty()) {
        const auto* current = impl->window(impl->capturedWindow);
        const auto next = std::find_if(config.windows.begin(), config.windows.end(), [&](const auto& item) {
            return item.id == impl->capturedWindow;
        });
        if (!current || next == config.windows.end() || current->content != next->content) {
            if (impl->pointerWindow == impl->capturedWindow) impl->pointerWindow.clear();
            impl->capturedWindow.clear();
            impl->capturedButtons.fill(false);
        }
    }
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    std::vector<std::string> next;
    for (const auto& id : impl->order) {
        if (impl->window(id)) {
            next.push_back(id);
        }
    }
    for (const auto& item : impl->config.windows) {
        if (std::find(next.begin(), next.end(), item.id) == next.end()) {
            next.push_back(item.id);
            impl->focused = item.id;
        }
    }
    for (const auto& item : impl->config.windows) {
        auto& chrome = impl->chrome[item.id];
        if (!chrome) {
            chrome = std::make_unique<Impl::Chrome>();
        }
        chrome->label.update(impl->titleConfig(item));
        chrome->close.update(impl->closeConfig(item.id));
    }
    impl->order = std::move(next);
    std::erase_if(impl->states, [this](const auto& entry) {
        return !impl->window(entry.first);
    });
    std::erase_if(impl->chrome, [this](const auto& entry) {
        return !impl->window(entry.first);
    });
    if (!impl->window(impl->dragTarget)) {
        impl->drag = Impl::Drag::None;
    }
    mountChildren(impl->parts());
}

bool FloatingLayer::covers(const Extent2D<F32>& point) const {
    return impl->drag != Impl::Drag::None || !impl->windowAt(point).empty();
}

void FloatingLayer::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 radius = Px(ctx, Rounding);
    const F32 stroke = Px(ctx, 1.0f);
    const F32 softness = Px(ctx, kShadowSoftness);
    const F32 drop = Px(ctx, kShadowDrop);
    const ColorRGBA<F32> clear = {0.0f, 0.0f, 0.0f, 0.0f};
    impl->titleHeight = std::round(Px(ctx, kFloatingTitleHeight));
    impl->titlePadX = Px(ctx, kFloatingTitlePadX);
    impl->closeButton = Px(ctx, kFloatingCloseButton);
    impl->closeGlyph = Px(ctx, kFloatingCloseGlyph);
    impl->grip = Px(ctx, kResizeGrip);
    impl->slack = Px(ctx, kResizeSlack);

    Context inner = ctx;
    inner.backdropColorKey = "tile_bg";
    inner.backdropTopColorKey.clear();

    for (const auto& id : impl->order) {
        const auto* item = impl->window(id);
        auto& chrome = *impl->chrome[id];
        auto& state = impl->states[id];
        if (!state.placed) {
            const Extent2D<F32> size = {Px(ctx, item->size.x), Px(ctx, item->size.y)};
            const F32 offset = Px(ctx, kFloatingCascade) * static_cast<F32>(impl->cascade++ % kFloatingCascadeSteps);
            state.rect = {std::round(bounds.x + (bounds.width - size.x) * 0.5f + offset),
                          std::round(bounds.y + (bounds.height - size.y) * 0.5f + offset), size.x, size.y};
            state.placed = true;
        }
        const F32 minimum = impl->titleHeight * 3.0f;
        state.rect.width = std::max(minimum, state.rect.width);
        state.rect.height = std::max(minimum, state.rect.height);
        state.rect.x = std::clamp(state.rect.x, bounds.x - state.rect.width + minimum, bounds.right() - minimum);
        state.rect.y = std::clamp(state.rect.y, bounds.y, std::max(bounds.y, bounds.bottom() - impl->titleHeight));

        const Rect rect = state.rect;
        const Rect shadow = Inflate({rect.x, rect.y + drop, rect.width, rect.height}, softness);
        chrome.shadow.update({
            .instances = {{.rect = shadow, .backgroundColor = ctx.color("tile_shadow")}},
            .cornerRadius = radius,
            .softness = softness,
        });
        chrome.card.update({
            .instances = {{.rect = rect, .backgroundColor = ctx.color("tile_bg")}},
            .cornerRadius = radius,
        });
        chrome.border.update({
            .instances = {{.rect = rect, .backgroundColor = clear}},
            .cornerRadius = radius,
            .borderWidth = stroke,
            .borderColor = ctx.color("tile_border"),
            .strokeFringe = true,
        });
        chrome.highlight.update({
            .instances = {{.rect = rect, .backgroundColor = clear}},
            .cornerRadius = radius,
            .borderWidth = stroke,
            .borderColor = ctx.color("tile_highlight"),
            .strokeFringe = true,
            .topLit = true,
        });
        chrome.rim.update({
            .instances = {{.rect = Inset(rect, stroke), .backgroundColor = clear}},
            .cornerRadius = std::max(0.0f, radius - stroke),
            .borderWidth = stroke,
            .borderColor = ctx.color("tile_rim"),
            .strokeFringe = true,
            .topLit = true,
        });
        chrome.label.update(impl->titleConfig(*item));
        chrome.close.update(impl->closeConfig(id));

        const Rect close = impl->closeRect(rect);
        const Extent2D<F32> glyph = measureChild(chrome.close, ctx, {infinite, infinite});
        const F32 titleSize = measureChild(chrome.label, ctx, {infinite, infinite}).y;
        const Rect title = {rect.x + impl->titlePadX, rect.y + std::round((impl->titleHeight - titleSize) * 0.5f),
                            std::max(0.0f, close.x - rect.x - impl->titlePadX), titleSize};

        layoutChild(ctx, chrome.shadow, shadow);
        layoutChild(ctx, chrome.card, rect);
        layoutChild(ctx, chrome.label, title, title);
        layoutChild(ctx, chrome.close,
                    {std::round(close.x + (close.width - glyph.x) * 0.5f),
                     std::round(close.y + (close.height - glyph.y) * 0.5f), glyph.x, glyph.y},
                    close);
        const Rect body = impl->bodyRect(rect);
        if (item->content) {
            layoutChild(inner, *item->content, Inset(body, Px(ctx, item->padding)), body);
        }
        layoutChild(ctx, chrome.border, rect);
        layoutChild(ctx, chrome.highlight, rect);
        layoutChild(ctx, chrome.rim, rect);
    }
}

bool FloatingLayer::deliverTo(const std::string& id, const MouseEvent& event) {
    const auto* item = impl->window(id);
    return item && item->content && deliverChild(*item->content, event);
}

void FloatingLayer::pointTo(const std::string& id, const MouseEvent& event) {
    if (impl->pointerWindow == id) {
        return;
    }
    MouseEvent leave = event;
    leave.type = MouseEventType::Leave;
    deliverTo(impl->pointerWindow, leave);
    impl->pointerWindow = id;
}

bool FloatingLayer::mouseEvent(const MouseEvent& event) {
    const Extent2D<F32> point = event.position;
    const U64 button = static_cast<U64>(event.button);
    if ((event.type == MouseEventType::Click || event.type == MouseEventType::Release) &&
        button >= impl->capturedButtons.size()) {
        return false;
    }
    switch (event.type) {
        case MouseEventType::Move: {
            if (impl->drag != Impl::Drag::None) {
                auto& state = impl->states[impl->dragTarget];
                const Extent2D<F32> delta = {point.x - impl->dragStart.x, point.y - impl->dragStart.y};
                const Rect before = state.rect;
                if (impl->drag == Impl::Drag::Move) {
                    state.rect.x = std::round(impl->dragRect.x + delta.x);
                    state.rect.y = std::round(impl->dragRect.y + delta.y);
                } else {
                    state.rect.width = std::round(impl->dragRect.width + delta.x);
                    state.rect.height = std::round(impl->dragRect.height + delta.y);
                }
                if (state.rect != before) {
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            const std::string target = impl->windowAt(point);
            const std::string closeHovered =
                !target.empty() && impl->closeRect(impl->states[target].rect).contains(point.x, point.y) ? target
                                                                                                        : std::string();
            if (closeHovered != impl->closeHovered) {
                impl->closeHovered = closeHovered;
                invalidate(Dirty::Layout);
            }
            const std::string receiver = impl->capturedWindow.empty() ? target : impl->capturedWindow;
            pointTo(receiver, event);
            deliverTo(receiver, event);
            return !target.empty();
        }
        case MouseEventType::Click: {
            if (impl->drag != Impl::Drag::None) {
                return true;
            }
            if (!impl->capturedWindow.empty()) {
                impl->capturedButtons[button] = true;
                deliverTo(impl->capturedWindow, event);
                return true;
            }
            const std::string target = impl->windowAt(point);
            if (target.empty()) {
                return false;
            }
            if (impl->focused != target || impl->order.back() != target) {
                impl->focused = target;
                impl->raise(target);
                mountChildren(impl->parts());
                if (const auto* item = impl->window(target); item && item->onFocus) {
                    item->onFocus();
                }
                invalidate(Dirty::Layout);
            }
            const Rect rect = impl->states[target].rect;
            if (event.button == MouseButton::Left) {
                if (impl->closeRect(rect).contains(point.x, point.y)) {
                    if (const auto* item = impl->window(target); item && item->onClose) {
                        item->onClose();
                    }
                    return true;
                }
                if (impl->gripRect(rect).contains(point.x, point.y)) {
                    impl->drag = Impl::Drag::Resize;
                } else if (impl->titleRect(rect).contains(point.x, point.y)) {
                    impl->drag = Impl::Drag::Move;
                }
                if (impl->drag != Impl::Drag::None) {
                    impl->dragTarget = target;
                    impl->dragStart = point;
                    impl->dragRect = rect;
                    return true;
                }
            }
            impl->capturedWindow = target;
            impl->capturedButtons[button] = true;
            pointTo(target, event);
            deliverTo(target, event);
            return true;
        }
        case MouseEventType::Release: {
            if (impl->drag != Impl::Drag::None) {
                if (event.button == MouseButton::Left) {
                    impl->drag = Impl::Drag::None;
                }
                return true;
            }
            const std::string target = impl->windowAt(point);
            const std::string receiver = impl->capturedWindow;
            impl->capturedButtons[button] = false;
            if (std::none_of(impl->capturedButtons.begin(), impl->capturedButtons.end(), [](bool held) { return held; })) {
                impl->capturedWindow.clear();
            }
            deliverTo(receiver.empty() ? target : receiver, event);
            return !target.empty();
        }
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            impl->capturedButtons.fill(false);
            impl->drag = Impl::Drag::None;
            if (!impl->closeHovered.empty()) {
                impl->closeHovered.clear();
                invalidate(Dirty::Layout);
            }
            if (impl->capturedWindow != impl->pointerWindow) {
                deliverTo(impl->capturedWindow, event);
            }
            deliverTo(impl->pointerWindow, event);
            impl->capturedWindow.clear();
            impl->pointerWindow.clear();
            return false;
        default: {
            const std::string target = impl->windowAt(point);
            const std::string receiver = impl->capturedWindow.empty() ? target : impl->capturedWindow;
            pointTo(receiver, event);
            deliverTo(receiver, event);
            return !target.empty();
        }
    }
}

std::optional<SurfaceCursor> FloatingLayer::cursorAt(const Extent2D<F32>& point) const {
    if (impl->drag == Impl::Drag::Resize) {
        return SurfaceCursor::ResizeNWSE;
    }
    const std::string target = impl->windowAt(point);
    if (!target.empty() && impl->gripRect(impl->states.at(target).rect).contains(point.x, point.y)) {
        return SurfaceCursor::ResizeNWSE;
    }
    return std::nullopt;
}

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
        CardSet chrome;
        Text icon;
        Text heading;
        Text content;
        Box closeBadge;
        Text closeGlyph;

        std::vector<Component*> parts() {
            std::vector<Component*> result = chrome.parts();
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

        const char* iconStr = ICON_FA_CIRCLE_INFO;
        const char* fallback = "Info";
        const char* colorKey = "info_blue";
        switch (item.type) {
            case ToastType::Success:
                iconStr = ICON_FA_CIRCLE_CHECK;
                fallback = "Success";
                colorKey = "success_green";
                break;
            case ToastType::Warning:
                iconStr = ICON_FA_TRIANGLE_EXCLAMATION;
                fallback = "Warning";
                colorKey = "warning_yellow";
                break;
            case ToastType::Error:
                iconStr = ICON_FA_CIRCLE_EXCLAMATION;
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

        item.chrome.update(ctx, {card}, bounds, radius, softness, Px(ctx, kToastShadowDrop), 1.0f, "notification_bg",
                           opacity);
        for (auto* part : item.chrome.parts()) {
            layoutChild(ctx, *part, bounds);
        }

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
        item.closeGlyph.update({.str = ICON_FA_XMARK,
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

struct BackgroundLayer::Impl {
    struct Particle {
        Extent2D<F32> position;
        F32 velocity = 0.0f;
        F32 radius = 0.0f;
        F32 alphaSeed = 0.0f;
        F32 phase = 0.0f;
    };

    Config config;
    Box fill;
    Box particles;
    std::vector<Particle> state;
    Rect area;
    std::mt19937 random{std::random_device{}()};
};

BackgroundLayer::BackgroundLayer() {
    impl = std::make_unique<Impl>();
    addChild(impl->fill);
    addChild(impl->particles);
}

BackgroundLayer::~BackgroundLayer() = default;

bool BackgroundLayer::covers(const Extent2D<F32>&) const {
    return false;
}

void BackgroundLayer::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    if (impl->config.particleCount != config.particleCount) {
        impl->state.clear();
    }
    impl->config = std::move(config);
}

void BackgroundLayer::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    const auto background = ctx.color(config.backgroundColorKey);
    impl->fill.update({.instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = background}}});
    layoutChild(ctx, impl->fill, bounds);

    const F32 top = Px(ctx, config.topOffset);
    const Rect area = {bounds.x, bounds.y + top, bounds.width, std::max(0.0f, bounds.height - top)};
    if (!config.particles || config.particleCount == 0 || area.empty()) {
        impl->particles.update({.instances = {}});
        layoutChild(ctx, impl->particles, bounds);
        return;
    }

    requestFrame();
    const F32 unit = ctx.pixelRatio;
    std::uniform_real_distribution<F32> uniform(0.0f, 1.0f);
    if (impl->state.size() != config.particleCount || impl->area != area) {
        impl->area = area;
        impl->state.clear();
        for (U64 index = 0; index < config.particleCount; ++index) {
            impl->state.push_back({
                .position = {area.x + uniform(impl->random) * area.width, area.y + uniform(impl->random) * area.height},
                .velocity = 10.0f + uniform(impl->random) * 30.0f,
                .radius = 0.8f + uniform(impl->random) * 1.8f,
                .alphaSeed = uniform(impl->random),
                .phase = uniform(impl->random) * 6.28318f,
            });
        }
    }

    const auto base = ctx.color(config.particleColorKey);
    const F32 luminance = 0.2126f * background.r + 0.7152f * background.g + 0.0722f * background.b;
    const bool dark = luminance < 0.5f;
    const F32 alphaLo = dark ? 0.2f : 0.28f;
    const F32 alphaHi = dark ? 0.6f : 0.58f;
    const F32 time = static_cast<F32>(ctx.time);
    const F32 delta = ctx.deltaTime;

    std::vector<Box::Instance> instances;
    instances.reserve(impl->state.size());
    for (auto& particle : impl->state) {
        particle.position.x += particle.velocity * 0.15f * delta * unit;
        particle.position.y -= particle.velocity * delta * unit;
        const F32 wave = std::sin(time * 0.5f + particle.phase) * 2.0f * unit;
        const Extent2D<F32> center = {particle.position.x + wave, particle.position.y};
        if (center.y < area.y - 10.0f * unit) {
            particle.position.y = area.bottom() + 10.0f * unit;
            particle.position.x = area.x + uniform(impl->random) * area.width;
        }
        if (center.x > area.right() + 10.0f * unit) {
            particle.position.x = area.x - 10.0f * unit;
        }
        const F32 twinkle = std::sin(time * 2.0f + particle.phase) * 0.1f + 0.9f;
        auto color = base;
        color.a = (alphaLo + particle.alphaSeed * (alphaHi - alphaLo)) * twinkle;
        const F32 radius = particle.radius * unit;
        instances.push_back({
            .rect = {center.x - radius, center.y - radius, 2.0f * radius, 2.0f * radius},
            .visible = center.y >= area.y,
            .backgroundColor = color,
        });
    }
    impl->particles.update({
        .instances = std::move(instances),
        .clip = area,
        .cornerRadius = 64.0f,
        .capacity = std::max<U64>(config.particleCount, 1),
    });
    layoutChild(ctx, impl->particles, bounds);
}

}  // namespace Jetstream::Sakura
