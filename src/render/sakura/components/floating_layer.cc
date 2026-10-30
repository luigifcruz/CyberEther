#include <jetstream/render/sakura/components/floating_layer.hh>

#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_fa.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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

}  // namespace

struct FloatingLayer::Impl {
    struct State {
        Rect rect;
        F32 pixelRatio = 1.0f;
        bool placed = false;
    };

    struct Chrome {
        Card base;
        Text label;
        Text close;
        Card frame;
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
            result.insert(result.end(), {&parts.base, &parts.label, &parts.close});
            if (item->content) {
                result.push_back(item->content);
            }
            result.push_back(&parts.frame);
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
            state.pixelRatio = ctx.pixelRatio;
            state.placed = true;
        }
        if (state.pixelRatio != ctx.pixelRatio && state.pixelRatio > 0.0f) {
            const F32 scale = ctx.pixelRatio / state.pixelRatio;
            state.rect = {std::round(state.rect.x * scale), std::round(state.rect.y * scale),
                          std::round(state.rect.width * scale), std::round(state.rect.height * scale)};
            state.pixelRatio = ctx.pixelRatio;
        }
        const F32 minimum = impl->titleHeight * 3.0f;
        state.rect.width = std::max(minimum, state.rect.width);
        state.rect.height = std::max(minimum, state.rect.height);
        state.rect.x = std::clamp(state.rect.x, bounds.x - state.rect.width + minimum, bounds.right() - minimum);
        state.rect.y = std::clamp(state.rect.y, bounds.y, std::max(bounds.y, bounds.bottom() - impl->titleHeight));

        const Rect rect = state.rect;
        Card::Config card = {
            .rects = {rect},
            .cornerRadius = radius,
            .softness = softness,
            .drop = drop,
            .borderWidth = stroke,
            .lineWidth = stroke,
            .layer = Card::Layer::Base,
        };
        chrome.base.update(card);
        card.layer = Card::Layer::Frame;
        chrome.frame.update(std::move(card));
        chrome.label.update(impl->titleConfig(*item));
        chrome.close.update(impl->closeConfig(id));

        const Rect close = impl->closeRect(rect);
        const Extent2D<F32> glyph = measureChild(chrome.close, ctx, {infinite, infinite});
        const F32 titleSize = measureChild(chrome.label, ctx, {infinite, infinite}).y;
        const Rect title = {rect.x + impl->titlePadX, rect.y + std::round((impl->titleHeight - titleSize) * 0.5f),
                            std::max(0.0f, close.x - rect.x - impl->titlePadX), titleSize};

        layoutChild(ctx, chrome.base, rect);
        layoutChild(ctx, chrome.label, title, title);
        layoutChild(ctx, chrome.close,
                    {std::round(close.x + (close.width - glyph.x) * 0.5f),
                     std::round(close.y + (close.height - glyph.y) * 0.5f), glyph.x, glyph.y},
                    close);
        const Rect body = impl->bodyRect(rect);
        if (item->content) {
            layoutChild(inner, *item->content, Inset(body, Px(ctx, item->padding)), body);
        }
        layoutChild(ctx, chrome.frame, rect);
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

}  // namespace Jetstream::Sakura
