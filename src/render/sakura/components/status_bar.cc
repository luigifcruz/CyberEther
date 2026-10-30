#include <jetstream/render/sakura/components/status_bar.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>

#include "../context.hh"
#include "../detail/text_metrics.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kHeight = 24.0f;
constexpr F32 kTextHorizontalPadding = 8.0f;
constexpr F32 kDotDiameter = 6.0f;
constexpr F32 kToggleInset = 4.0f;
constexpr F32 kToggleGap = 6.0f;
constexpr F32 kCaretRows = 4.0f;

ColorRGBA<F32> BarColor(const Context& ctx, const StatusTone tone) {
    switch (tone) {
        case StatusTone::Success: return ctx.color("editor_status_success");
        case StatusTone::Warning: return ctx.color("editor_status_warning");
        case StatusTone::Error: return ctx.color("editor_status_error");
        case StatusTone::Info: break;
    }
    return ctx.color("editor_status_info");
}

ColorRGBA<F32> TextColor(const Context& ctx, const StatusTone tone) {
    switch (tone) {
        case StatusTone::Success: return ctx.color("editor_status_success_text");
        case StatusTone::Warning: return ctx.color("editor_status_warning_text");
        case StatusTone::Error: return ctx.color("editor_status_error_text");
        case StatusTone::Info: break;
    }
    return ctx.color("editor_status_info_text");
}

ColorRGBA<F32> SeparatorColor(const Context& ctx, const StatusTone tone) {
    switch (tone) {
        case StatusTone::Success: return ctx.color("editor_status_success_separator");
        case StatusTone::Warning: return ctx.color("editor_status_warning_separator");
        case StatusTone::Error: return ctx.color("editor_status_error_separator");
        case StatusTone::Info: break;
    }
    return ctx.color("editor_status_info_separator");
}

}  // namespace

struct StatusBar::Impl {
    Config config;

    Box bar;
    Box pill;
    Box dot;
    Box caret;
    Label labels;

    TextMetrics textMetrics;
    Rect toggleRect;
};

Extent2D<F32> StatusBar::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(std::max(kHeight, kHeight * ctx.pixelRatio), available.y)};
}

StatusBar::StatusBar() {
    this->impl = std::make_unique<Impl>();
    addChild(this->impl->bar);
    addChild(this->impl->pill);
    addChild(this->impl->dot);
    addChild(this->impl->caret);
    addChild(this->impl->labels);
}

StatusBar::~StatusBar() = default;

void StatusBar::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    this->impl->config = std::move(config);
}

void StatusBar::layout(const Context& ctx) {
    auto& impl = *this->impl;
    const auto& config = impl.config;
    impl.textMetrics.setWindow(ctx.render);

    const Rect bar = frame();
    const F32 ratio = std::max(0.1f, ctx.pixelRatio);
    const bool visible = !bar.empty() && (!config.status.empty() || !config.toggleText.empty());
    const bool statusTextOn = visible && !config.status.empty();
    const bool toggleOn = visible && !config.toggleText.empty();

    const F32 pad = kTextHorizontalPadding * ratio;
    const F32 gap = kToggleGap * ratio;
    const F32 inset = kToggleInset * ratio;
    const F32 outline = std::max(1.0f, std::round(ratio));
    const F32 dot = std::max(2.0f, std::round(kDotDiameter * ratio));
    const F32 caretRows = std::max(2.0f, std::round(kCaretRows * ratio));
    const F32 caretWidth = 2.0f * caretRows - 1.0f;
    const F32 fontSize = config.fontSize * Typography::ChromeScale;
    const auto textColor = TextColor(ctx, config.tone);
    const auto separatorColor = SeparatorColor(ctx, config.tone);

    impl.toggleRect = {};
    if (toggleOn) {
        const F32 textWidth = impl.textMetrics.measure("default_mono", config.toggleText, fontSize);
        const F32 width = std::round(textWidth + gap + caretWidth + 2.0f * pad);
        impl.toggleRect = {std::round(bar.right() - pad - width),
                           std::round(bar.y + inset),
                           width,
                           std::max(0.0f, std::round(bar.height - 2.0f * inset))};
    }
    const auto& toggle = impl.toggleRect;
    const F32 messageLeft = bar.x + pad + dot + gap;
    const F32 messageRight = toggleOn ? toggle.x - gap : bar.right() - pad;

    impl.bar.update({
        .instances = {
            {.rect = bar, .visible = visible, .backgroundColor = BarColor(ctx, config.tone),
             .corners = Box::BottomLeft | Box::BottomRight},
            {.rect = {bar.x, std::floor(bar.y), bar.width, outline},
             .visible = visible, .backgroundColor = separatorColor,
             .corners = Box::BottomLeft | Box::BottomRight},
        },
        .cornerRadius = config.cornerRadius,
    });
    impl.dot.update({
        .instances = {{
            .rect = {std::round(bar.x + pad), std::round(bar.y + (bar.height - dot) * 0.5f), dot, dot},
            .visible = statusTextOn,
            .backgroundColor = textColor,
        }},
        .cornerRadius = dot * 0.5f,
    });
    impl.pill.update({
        .instances = {{
            .rect = toggle,
            .visible = toggleOn,
            .backgroundColor = separatorColor,
        }},
        .cornerRadius = toggle.height * 0.5f,
    });

    const F32 caretX = toggle.right() - pad - caretWidth;
    const F32 caretY = std::round(toggle.y + (toggle.height - caretRows) * 0.5f);
    std::vector<Box::Instance> caretInstances;
    caretInstances.reserve(static_cast<U64>(caretRows));
    for (F32 row = 0.0f; row < caretRows; row += 1.0f) {
        const F32 step = config.toggleExpanded ? caretRows - 1.0f - row : row;
        const F32 width = 2.0f * step + 1.0f;
        caretInstances.push_back({
            .rect = {caretX + (caretWidth - width) * 0.5f, caretY + row, width, 1.0f},
            .visible = toggleOn,
            .backgroundColor = textColor,
        });
    }
    impl.caret.update({
        .instances = std::move(caretInstances),
    });

    const F32 messageWidth = std::max(0.0f, messageRight - messageLeft);
    std::string message = impl.textMetrics.ellipsize("default_mono", config.status, fontSize, messageWidth);
    const bool messageFits = impl.textMetrics.measure("default_mono", message, fontSize) <= messageWidth;
    impl.labels.update({
        .instances = {
            {.rect = {messageLeft, bar.y, messageWidth, bar.height},
             .str = std::move(message),
             .visible = statusTextOn && messageFits,
             .color = textColor,
             .fontSize = fontSize,
             .alignment = {0, 1}},
            {.rect = {toggle.x + pad, toggle.y, std::max(0.0f, caretX - gap - (toggle.x + pad)), toggle.height},
             .str = config.toggleText,
             .visible = toggleOn,
             .color = textColor,
             .fontSize = fontSize,
             .alignment = {0, 1}},
        },
        .fontName = "default_mono",
    });

    layoutChild(ctx, impl.bar, bar);
    layoutChild(ctx, impl.pill, bar);
    layoutChild(ctx, impl.dot, bar);
    layoutChild(ctx, impl.caret, bar);
    layoutChild(ctx, impl.labels, bar);
}

std::optional<SurfaceCursor> StatusBar::cursorAt(const Extent2D<F32>& point) const {
    const auto& impl = *this->impl;
    if (impl.toggleRect.empty() || !impl.config.onToggle || !impl.toggleRect.contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

bool StatusBar::mouseEvent(const MouseEvent& event) {
    const auto& impl = *this->impl;
    if (impl.toggleRect.empty() || !impl.config.onToggle) {
        return false;
    }
    const bool inside = impl.toggleRect.contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Click:
            if (event.button == MouseButton::Left && inside) {
                impl.config.onToggle();
                return true;
            }
            return false;
        case MouseEventType::Move:
            return false;
        default:
            return false;
    }
}

}  // namespace Jetstream::Sakura
