#include <jetstream/render/sakura/components/pill_field.hh>

#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_input.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kPillPadX = 14.0f;
constexpr F32 kPillIcon = 13.0f;

}  // namespace

struct PillField::Impl {
    Config config;
    Card card;
    Text icon;
    Text placeholder;
    Text trailing;
    TextInput input;
    bool pressed = false;
    Rect bounds;
    std::vector<Component*> mounted;

    void mount(PillField& self) {
        std::vector<Component*> children = {&card};
        children.push_back(&icon);
        if (config.editing) {
            children.push_back(&input);
        } else {
            children.push_back(&placeholder);
        }
        if (!config.trailing.empty()) {
            children.push_back(&trailing);
        }
        if (children != mounted) {
            self.mountChildren(children);
            mounted = std::move(children);
        }
    }
};

PillField::PillField() {
    impl = std::make_unique<Impl>();
    setClipsChildren(false);
}

PillField::~PillField() = default;

void PillField::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->icon.update({.str = impl->config.icon, .colorKey = impl->config.iconColorKey,
                       .scale = kPillIcon / Typography::FontSize});
    impl->placeholder.update({.str = impl->config.placeholder, .font = Text::Font::Display, .tone = Text::Tone::Secondary});
    impl->trailing.update({.str = impl->config.trailing, .tone = Text::Tone::Secondary, .scale = 0.85f});
    impl->input.update({
        .value = impl->config.query,
        .hint = impl->config.placeholder,
        .submit = TextInput::Submit::OnEdit,
        .focus = impl->config.editing,
        .onChange = impl->config.onChange,
        .onSubmit = impl->config.onSubmit,
        .backgroundColorKey = "transparent",
        .onCancel = impl->config.onCancel,
        .padding = {0.0f, 0.0f},
        .fontName = Typography::DisplayFont,
    });
    impl->mount(*this);
}

Extent2D<F32> PillField::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& size = impl->config.size;
    const F32 height = ResolveLength(ctx, size.y, Px(ctx, kPillHeight), available.y);
    if (size.x > 0.0f) {
        return {ResolveLength(ctx, size.x, 0.0f, available.x), height};
    }
    const F32 infinite = std::numeric_limits<F32>::infinity();
    F32 width = 2.0f * Px(ctx, kPillPadX) + measureChild(impl->placeholder, ctx, {infinite, infinite}).x;
    if (!impl->config.icon.empty()) {
        width += measureChild(impl->icon, ctx, {infinite, infinite}).x + Px(ctx, kPillIconGap);
    }
    if (!impl->config.trailing.empty()) {
        width += measureChild(impl->trailing, ctx, {infinite, infinite}).x + Px(ctx, kPillIconGap);
    }
    return {ResolveLength(ctx, size.x, std::ceil(width), available.x), height};
}

void PillField::layout(const Context& ctx) {
    const Rect bounds = frame();
    impl->bounds = bounds;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 radius = bounds.height * 0.5f;
    impl->card.update({
        .rects = {bounds},
        .clip = clip(),
        .cornerRadius = radius,
        .softness = Px(ctx, kPillShadowSoftness),
        .drop = Px(ctx, kPillShadowDrop),
        .borderWidth = Px(ctx, kPillBorder),
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = kPillFill,
    });
    layoutChild(ctx, impl->card, bounds);

    F32 x = bounds.x + Px(ctx, kPillPadX);
    F32 right = bounds.right() - Px(ctx, kPillPadX);
    if (!impl->config.icon.empty()) {
        const Extent2D<F32> size = measureChild(impl->icon, ctx, {infinite, infinite});
        layoutChild(ctx, impl->icon, {x, bounds.y + (bounds.height - size.y) * 0.5f, size.x, size.y});
        x += size.x + Px(ctx, kPillIconGap);
    }
    if (!impl->config.trailing.empty()) {
        const Extent2D<F32> size = measureChild(impl->trailing, ctx, {infinite, infinite});
        layoutChild(ctx, impl->trailing, {right - size.x, bounds.y + (bounds.height - size.y) * 0.5f, size.x, size.y});
        right -= size.x + Px(ctx, kPillIconGap);
    }
    const Rect text = {x, bounds.y, std::max(0.0f, right - x), bounds.height};
    if (impl->config.editing) {
        layoutChild(ctx, impl->input, text);
    } else {
        const F32 height = measureChild(impl->placeholder, ctx, {infinite, infinite}).y;
        layoutChild(ctx, impl->placeholder, {text.x, bounds.y + (bounds.height - height) * 0.5f, text.width, height},
                    text);
    }
}

bool PillField::mouseEvent(const MouseEvent& event) {
    if (impl->config.editing) {
        return mouseEventChildren(event);
    }
    const bool inside = impl->bounds.contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Click:
            if (event.button == MouseButton::Left) {
                impl->pressed = inside;
            }
            return inside;
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left) {
                return false;
            }
            const bool pressed = impl->pressed;
            impl->pressed = false;
            if (pressed && inside && impl->config.onOpen) {
                impl->config.onOpen();
            }
            return pressed;
        }
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            impl->pressed = false;
            return false;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> PillField::cursorAt(const Extent2D<F32>& point) const {
    if (!impl->config.editing && impl->bounds.contains(point.x, point.y)) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
