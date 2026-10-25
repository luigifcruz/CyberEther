#include <jetstream/render/sakura/components/chat_composer.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/model_picker.hh>
#include <jetstream/render/sakura/components/popup.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/popup_layer.hh"
#include "../detail/text_metrics.hh"
#include "tools/text.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr std::array<F32, 5> kVoiceBarHeights = {0.45f, 1.0f, 0.6f, 0.85f, 0.3f};
constexpr F32 kVoiceIconHeight = 0.44f;
constexpr F32 kVoiceBarWidth = 0.075f;
constexpr F32 kVoiceBarGap = 0.06f;
constexpr U64 kUsageRingDots = 48;
constexpr F32 kUsageRingRadius = 0.42f;
constexpr F32 kUsageRingDot = 0.13f;
constexpr F32 kUsageRingTrackAlpha = 0.35f;
constexpr F32 kTooltipPaddingX = 14.0f;
constexpr F32 kTooltipPaddingY = 8.0f;
constexpr F32 kTooltipGap = 6.0f;
constexpr F32 kTooltipRounding = 14.0f;
constexpr F32 kTooltipLineScale = 1.15f;
constexpr U64 kTooltipLines = 3;
constexpr F32 kControlPadding = 14.0f;
constexpr F32 kButtonPaddingY = 4.0f;
constexpr F32 kIconWidth = 1.8f;
constexpr F32 kIconScale = 1.1f;
constexpr F32 kAttachScale = 0.9f;
constexpr F32 kPickerRounding = 15.0f;
constexpr U64 kMaxCharacters = 128;

struct RoundButton : public Component {
    struct Config {
        std::string str;
        std::string fontName = Typography::IconFont;
        F32 fontSize = Typography::FontSize;
        F32 padding = 0.0f;
        F32 rounding = 0.0f;
        bool disabled = false;
        std::string colorKey = "transparent";
        std::string hoveredColorKey = "button_hovered";
        std::string activeColorKey = "button_active";
        std::string textColorKey = "text_primary";
        Callback<void()> onClick;

        bool operator==(const Config&) const = default;
    };

    Config config;
    Box background;
    Label label;
    TextMetrics metrics;
    bool hovered = false;
    bool pressed = false;

    RoundButton() {
        addChild(background);
        addChild(label);
    }

    void update(Config next) {
        if (next != config) {
            invalidate(Dirty::Layout);
        }
        config = std::move(next);
        if (config.disabled && (hovered || pressed)) {
            hovered = false;
            pressed = false;
            invalidate(Dirty::Layout);
        }
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        metrics.setWindow(ctx.render);
        const F32 text = config.str.empty() ? 0.0f : metrics.measure(config.fontName, config.str, Px(ctx, config.fontSize));
        return FitExtent({std::ceil(text + 2.0f * Px(ctx, config.padding)),
                          Px(ctx, config.fontSize + 2.0f * kButtonPaddingY)},
                         available);
    }

    void layout(const Context& ctx) override {
        const Rect bounds = frame();
        const F32 alpha = config.disabled ? kDisabledAlpha : 1.0f;
        const std::string& fill = config.disabled      ? config.colorKey
                                  : pressed && hovered ? config.activeColorKey
                                  : hovered            ? config.hoveredColorKey
                                                       : config.colorKey;
        background.update({
            .instances = {{.rect = bounds, .visible = !bounds.empty(),
                           .backgroundColor = WithAlpha(ctx.color(fill), alpha)}},
            .clip = bounds,
            .cornerRadius = Px(ctx, config.rounding),
        });
        label.update({
            .instances = {{.rect = bounds, .str = config.str, .visible = !bounds.empty() && !config.str.empty(),
                           .color = WithAlpha(ctx.color(config.textColorKey), alpha),
                           .fontSize = Px(ctx, config.fontSize), .alignment = {1, 1}}},
            .clip = bounds,
            .fontName = config.fontName,
            .maxCharacters = 16,
        });
        layoutChild(ctx, background, bounds);
        layoutChild(ctx, label, bounds);
    }

    bool mouseEvent(const MouseEvent& event) override {
        if (config.disabled) {
            return false;
        }
        const bool inside = frame().contains(event.position.x, event.position.y);
        switch (event.type) {
            case MouseEventType::Move:
                if (inside != hovered) {
                    hovered = inside;
                    invalidate(Dirty::Layout);
                }
                return false;
            case MouseEventType::Leave:
            case MouseEventType::Cancel:
                if (hovered || pressed) {
                    hovered = false;
                    pressed = false;
                    invalidate(Dirty::Layout);
                }
                return false;
            case MouseEventType::Click:
                if (event.button != MouseButton::Left || !inside) {
                    return false;
                }
                pressed = true;
                hovered = true;
                invalidate(Dirty::Layout);
                return true;
            case MouseEventType::Release:
                if (event.button != MouseButton::Left || !pressed) {
                    return false;
                }
                pressed = false;
                invalidate(Dirty::Layout);
                if (inside && config.onClick) {
                    config.onClick();
                }
                return true;
            default:
                return false;
        }
    }

    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override {
        if (config.disabled || !frame().contains(point.x, point.y)) {
            return std::nullopt;
        }
        return SurfaceCursor::Hand;
    }
};

struct UsageTooltip : public Component {
    struct Config {
        std::vector<std::string> lines;
        F32 fontSize = Typography::FontSize;

        bool operator==(const Config&) const = default;
    };

    Config config;
    Box background;
    Label texts;
    TextMetrics metrics;

    UsageTooltip() {
        addChild(background);
        addChild(texts);
    }

    void update(Config next) {
        if (next != config) {
            invalidate(Dirty::Layout);
        }
        config = std::move(next);
    }

    F32 unit(const Context& ctx) const {
        return Px(ctx, config.fontSize / Typography::FontSize);
    }

    F32 lineHeight(const Context& ctx) const {
        return Px(ctx, config.fontSize) * Typography::CodeLineHeight * kTooltipLineScale;
    }

    Extent2D<F32> extent(const Context& ctx) {
        metrics.setWindow(ctx.render);
        F32 width = 0.0f;
        for (const auto& line : config.lines) {
            width = std::max(width, metrics.measure(Typography::BodyFont, line, Px(ctx, config.fontSize)));
        }
        return {std::ceil(width + 2.0f * kTooltipPaddingX * unit(ctx)),
                std::ceil(static_cast<F32>(config.lines.size()) * lineHeight(ctx) + 2.0f * kTooltipPaddingY * unit(ctx))};
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        return FitExtent(extent(ctx), available);
    }

    void layout(const Context& ctx) override {
        const Rect bounds = frame();
        background.update({
            .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color("popup_bg")}},
            .clip = bounds,
            .cornerRadius = kTooltipRounding * unit(ctx),
            .borderWidth = Px(ctx, 1.0f),
            .borderColor = ctx.color("border"),
            .strokeFringe = true,
        });
        const F32 line = lineHeight(ctx);
        std::vector<Label::Instance> instances;
        for (U64 i = 0; i < config.lines.size() && i < kTooltipLines; ++i) {
            instances.push_back({
                .rect = {bounds.x, bounds.y + kTooltipPaddingY * unit(ctx) + static_cast<F32>(i) * line, bounds.width,
                         line},
                .str = config.lines[i],
                .color = ctx.color(i == 0 ? "text_secondary" : "text_primary"),
                .fontSize = Px(ctx, config.fontSize),
                .alignment = {1, 1},
            });
        }
        texts.update({
            .instances = std::move(instances),
            .clip = bounds,
            .fontName = Typography::BodyFont,
            .maxCharacters = kMaxCharacters,
            .capacity = kTooltipLines,
        });
        layoutChild(ctx, background, bounds);
        layoutChild(ctx, texts, bounds);
    }
};

}  // namespace

struct ChatComposer::Impl {
    struct Metrics {
        F32 unit = 1.0f;
        F32 font = 0.0f;
        F32 lineHeight = 0.0f;
        F32 pad = 0.0f;
        F32 textPad = 0.0f;
        F32 control = 0.0f;
        F32 iconWidth = 0.0f;
    };

    struct Rects {
        Rect input;
        Rect send;
    };

    Config config;
    Box background;
    Label placeholder;
    Label icon;
    TextGrid input;
    RoundButton send;
    Box voiceBars;
    RoundButton attach;
    RoundButton clear;
    ModelPicker picker;
    Box usageRing;
    UsageTooltip tooltip;
    std::vector<Component*> mounted;
    std::string text;
    Rect usageRect;
    bool usageHovered = false;
    bool stale = true;
    F32 syncedRatio = 0.0f;

    F32 scale() const {
        return config.style.fontSize / Typography::FontSize;
    }

    F32 controlSize() const {
        return config.style.fontSize + kControlPadding * scale();
    }

    Metrics metrics(const Context& ctx) const {
        const auto& style = config.style;
        const F32 unit = Px(ctx, scale());
        const F32 font = Px(ctx, style.fontSize);
        return {
            .unit = unit,
            .font = font,
            .lineHeight = font * Typography::CodeLineHeight,
            .pad = Px(ctx, style.padding),
            .textPad = font * Typography::TextPaddingRatio,
            .control = Px(ctx, controlSize()),
            .iconWidth = style.icon.empty() ? 0.0f : font * kIconWidth,
        };
    }

    bool showsControls() const {
        return !config.style.compact;
    }

    bool showsClear() const {
        return showsControls() && config.style.clearButton && config.onClear;
    }

    bool showsPicker() const {
        return showsControls() && config.selector.has_value();
    }

    bool showsUsage() const {
        return showsControls() && config.usage.has_value();
    }

    bool voiceReady() const {
        return !config.busy && config.onVoice && Jetstream::Text::IsBlank(text);
    }

    std::vector<std::string> tooltipLines() const {
        std::vector<std::string> lines;
        if (!config.usage) {
            return lines;
        }
        lines.push_back(config.usage->title);
        for (const auto& detail : config.usage->details) {
            if (lines.size() == kTooltipLines) {
                break;
            }
            lines.push_back(detail);
        }
        return lines;
    }

    Rect inset(const Context& ctx, const Rect& outer) const {
        const auto& margin = config.style.margin;
        const F32 left = Px(ctx, margin.left);
        const F32 top = Px(ctx, margin.top);
        return {outer.x + left, outer.y + top, std::max(0.0f, outer.width - left - Px(ctx, margin.right)),
                std::max(0.0f, outer.height - top - Px(ctx, margin.bottom))};
    }

    F32 inputWidth(const Metrics& m, F32 boxWidth) const {
        F32 width = std::max(0.0f, boxWidth - 2.0f * m.pad - m.iconWidth);
        if (config.style.compact) {
            width = std::max(0.0f, width - m.control - m.textPad);
        }
        return width;
    }

    Rects rects(const Metrics& m, const Rect& bounds, F32 textHeight) const {
        const F32 offset = config.style.compact ? std::max(0.0f, (bounds.height - 2.0f * m.pad - textHeight) * 0.5f)
                                                : 0.0f;
        const F32 x = bounds.x + m.pad + m.iconWidth;
        const F32 y = bounds.y + m.pad + offset + m.textPad;
        if (config.style.compact) {
            const Rect sendRect = {bounds.right() - m.pad - m.control, bounds.bottom() - m.pad - m.control, m.control,
                                   m.control};
            return {
                .input = {x, y, std::max(0.0f, sendRect.x - m.textPad - x),
                          std::max(0.0f, bounds.bottom() - m.pad - offset - m.textPad - y)},
                .send = sendRect,
            };
        }
        const Rect inputRect = {x, y, std::max(0.0f, bounds.width - 2.0f * m.pad - m.iconWidth),
                                std::max(0.0f, bounds.bottom() - m.pad - m.control - m.textPad - y)};
        return {
            .input = inputRect,
            .send = {bounds.right() - m.pad - m.control, inputRect.bottom() + m.textPad, m.control, m.control},
        };
    }

    void edit(ChatComposer& self, std::string value) {
        if (value == text) {
            return;
        }
        text = std::move(value);
        stale = true;
        self.invalidate(Dirty::Layout);
    }

    void submit() {
        if (config.busy || Jetstream::Text::IsBlank(text)) {
            return;
        }
        if (config.onSubmit) {
            config.onSubmit(text);
        }
    }

    void press() {
        if (config.busy) {
            if (config.onCancel) {
                config.onCancel();
            }
            return;
        }
        if (voiceReady()) {
            config.onVoice();
            return;
        }
        submit();
    }

    void sync(ChatComposer& self, const Context& ctx) {
        if (!stale && syncedRatio == ctx.pixelRatio) {
            return;
        }
        stale = false;
        syncedRatio = ctx.pixelRatio;
        const auto m = metrics(ctx);
        input.update({
            .id = config.id + ":input",
            .value = text,
            .editable = true,
            .fontSize = m.font,
            .fontName = Typography::BodyFont,
            .monospace = false,
            .lineNumbers = false,
            .showActiveLine = false,
            .scrollbar = true,
            .scrollPastEnd = false,
            .minLines = config.style.minLines,
            .maxLines = config.style.maxLines,
            .wrap = TextGrid::Wrap::Word,
            .padding = Padding{m.textPad, 0.0f, m.textPad, 0.0f},
            .textColorKey = "editor_text",
            .selectionColorKey = "editor_selection",
            .selectionMatchColorKey = "editor_selection_match",
            .cursorColorKey = "editor_cursor",
            .scrollbarTrackColorKey = "editor_scrollbar_track",
            .scrollbarThumbColorKey = "editor_scrollbar_thumb",
            .submitOnEnter = true,
            .onChange = [this, &self](std::string value) {
                edit(self, std::move(value));
            },
            .onSubmit = [this, &self](std::string value) {
                edit(self, std::move(value));
                submit();
            },
            .onFocusChange = [this](bool focused) {
                if (config.onFocusChange) {
                    config.onFocusChange(focused);
                }
            },
        });
    }

    void mount(ChatComposer& self) {
        std::vector<Component*> list = {&background, &placeholder, &icon, &input, &send, &voiceBars};
        if (showsControls()) {
            list.push_back(&attach);
        }
        if (showsClear()) {
            list.push_back(&clear);
        }
        if (showsPicker()) {
            list.push_back(&picker);
        }
        list.push_back(&usageRing);
        if (list != mounted) {
            mounted = list;
            self.mountChildren(list);
        }
    }

    void configure() {
        const F32 control = controlSize();
        attach.update({
            .str = ICON_LC_PLUS,
            .fontSize = config.style.fontSize * kAttachScale,
            .rounding = 0.5f * control,
            .onClick = [this]() {
                if (config.onAttach) {
                    config.onAttach();
                }
            },
        });
        clear.update({
            .str = "Clear",
            .fontName = Typography::BodyFont,
            .fontSize = config.style.fontSize,
            .padding = 0.5f * control,
            .rounding = 0.5f * control,
            .textColorKey = "text_secondary",
            .onClick = [this]() {
                if (config.onClear) {
                    config.onClear();
                }
            },
        });
        const Selector selector = config.selector.value_or(Selector{});
        picker.update({
            .models = selector.options,
            .value = selector.value,
            .placeholder = selector.placeholder,
            .efforts = selector.efforts,
            .effort = selector.effort,
            .disabled = selector.disabled,
            .fontSize = config.style.fontSize,
            .rounding = 0.5f * control,
            .popupRounding = kPickerRounding * scale(),
            .accentColorKey = "agent_activity",
            .popupColorKey = config.style.backgroundColorKey,
            .onSelect = [this](const std::string& value) {
                if (config.selector && config.selector->onSelect) {
                    config.selector->onSelect(value);
                }
            },
            .onEffort = [this](const std::string& value) {
                if (config.selector && config.selector->onEffort) {
                    config.selector->onEffort(value);
                }
            },
        });
        tooltip.update({.lines = tooltipLines(), .fontSize = config.style.fontSize});
    }

    void layoutControls(ChatComposer& self, const Context& ctx, const Metrics& m, const Rect& outer,
                        const Rect& bounds, const Rect& sendRect) {
        const F32 y = sendRect.y;
        const Rect attachRect = {bounds.x + m.pad, y, m.control, m.control};
        self.layoutChild(ctx, attach, attachRect);

        const F32 clearWidth = showsClear() ? self.measureChild(clear, ctx, {bounds.width, m.control}).x : 0.0f;
        const Rect clearRect = {attachRect.right(), y, clearWidth, m.control};
        if (showsClear()) {
            self.layoutChild(ctx, clear, clearRect);
        }

        const F32 gap = m.textPad;
        const F32 ring = showsUsage() ? m.control : 0.0f;
        const F32 pickerRight = sendRect.x - gap;
        const F32 limit = std::max(0.0f, pickerRight - (clearRect.right() + gap) - ring);
        const F32 pickerWidth = showsPicker() ? std::min(limit, self.measureChild(picker, ctx, {limit, m.control}).x)
                                              : 0.0f;
        const Rect pickerRect = {pickerRight - pickerWidth, y, pickerWidth, m.control};
        if (showsPicker()) {
            self.layoutChild(ctx, picker, pickerRect);
        }

        usageRect = showsUsage() ? Rect{pickerRect.x - m.control, y, m.control, m.control} : Rect{};
        layoutUsage(self, ctx, m, outer);
    }

    void layoutUsage(ChatComposer& self, const Context& ctx, const Metrics& m, const Rect& outer) {
        const bool shown = showsUsage() && !usageRect.empty();
        const F32 used = config.usage ? std::clamp(config.usage->value, 0.0f, 1.0f) : 0.0f;
        const F32 radius = m.font * kUsageRingRadius;
        const F32 dot = m.font * kUsageRingDot;
        const auto center = usageRect.center();
        const auto fill = ctx.color("text_primary");
        const auto track = WithAlpha(ctx.color("text_secondary"), kUsageRingTrackAlpha);
        std::vector<Box::Instance> dots;
        for (U64 i = 0; i < kUsageRingDots; ++i) {
            const F32 position = static_cast<F32>(i) / static_cast<F32>(kUsageRingDots);
            const F32 angle = (position * 2.0f - 0.5f) * std::numbers::pi_v<F32>;
            dots.push_back({
                .rect = {center.x + radius * std::cos(angle) - dot * 0.5f, center.y + radius * std::sin(angle) - dot * 0.5f,
                         dot, dot},
                .visible = shown,
                .backgroundColor = position < used ? fill : track,
            });
        }
        usageRing.update({
            .instances = std::move(dots),
            .clip = usageRect,
            .cornerRadius = dot * 0.5f,
            .capacity = kUsageRingDots,
        });
        self.layoutChild(ctx, usageRing, usageRect);

        if (!shown || !usageHovered) {
            return;
        }
        const auto size = tooltip.extent(ctx);
        const F32 x = std::clamp(center.x - size.x * 0.5f, outer.x, std::max(outer.x, outer.right() - size.x));
        RequestPopup(ctx, {
            .content = &tooltip,
            .anchor = {std::round(x), std::round(usageRect.y - kTooltipGap * m.unit - size.y), 0.0f, 0.0f},
            .placement = PopupPlacement::Point,
            .interactive = false,
        });
    }
};

ChatComposer::ChatComposer() {
    impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->placeholder);
    addChild(impl->icon);
    addChild(impl->input);
    addChild(impl->send);
    addChild(impl->voiceBars);
    addChild(impl->attach);
    addChild(impl->clear);
    addChild(impl->picker);
    addChild(impl->usageRing);
}

ChatComposer::~ChatComposer() = default;

void ChatComposer::update(Config config) {
    auto& state = *impl;
    if (config != state.config) {
        invalidate(Dirty::Layout);
    }
    const auto& previous = state.config;
    if (config.focusRequest != previous.focusRequest) {
        if (textEditorFocused()) {
            state.input.focus();
        }
        state.input.offerFocus(true);
    }
    if (config.clearRequest != previous.clearRequest && !state.text.empty()) {
        state.text.clear();
        state.stale = true;
        invalidate(Dirty::Layout);
    }
    state.stale = state.stale || config.id != previous.id || config.style.fontSize != previous.style.fontSize ||
                  config.style.minLines != previous.style.minLines || config.style.maxLines != previous.style.maxLines;
    state.config = std::move(config);
    if (!state.showsUsage() && state.usageHovered) {
        state.usageHovered = false;
    }
    state.mount(*this);
    state.configure();
}

Extent2D<F32> ChatComposer::measure(const Context& ctx, Extent2D<F32> available) {
    auto& state = *impl;
    const auto& style = state.config.style;
    state.sync(*this, ctx);
    const auto m = state.metrics(ctx);
    const F32 width = FillLength(available.x, Px(ctx, kNaturalControlWidth));
    const F32 boxWidth = std::max(0.0f, width - Px(ctx, style.margin.left + style.margin.right));
    const F32 textHeight = measureChild(state.input, ctx,
                                        {state.inputWidth(m, boxWidth), std::numeric_limits<F32>::infinity()}).y +
                           2.0f * m.textPad;
    const F32 boxHeight = style.compact ? std::max(textHeight, m.control) + 2.0f * m.pad
                                        : textHeight + m.control + 2.0f * m.pad;
    return FitExtent({width, std::ceil(boxHeight + Px(ctx, style.margin.top + style.margin.bottom))}, available);
}

void ChatComposer::layout(const Context& ctx) {
    auto& state = *impl;
    const auto& config = state.config;
    const auto& style = config.style;
    state.sync(*this, ctx);
    const auto m = state.metrics(ctx);
    const Rect outer = frame();
    const Rect bounds = state.inset(ctx, outer);
    const bool visible = !bounds.empty();

    state.background.update({
        .instances = {{.rect = bounds, .visible = visible, .backgroundColor = ctx.color(style.backgroundColorKey)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, style.rounding.value_or(style.padding + 0.5f * state.controlSize())),
        .borderWidth = Px(ctx, style.borderWidth),
        .borderColor = ctx.color(style.borderColorKey),
        .strokeFringe = true,
    });
    layoutChild(ctx, state.background, bounds);

    const F32 measured = measureChild(state.input, ctx,
                                      {state.inputWidth(m, bounds.width), std::numeric_limits<F32>::infinity()}).y;
    const auto rects = state.rects(m, bounds, measured + 2.0f * m.textPad);

    state.placeholder.update({
        .instances = {{
            .rect = {rects.input.x + m.textPad, rects.input.y, std::max(0.0f, rects.input.width - 2.0f * m.textPad),
                     m.lineHeight},
            .str = style.placeholder,
            .visible = visible && !style.placeholder.empty() && state.text.empty(),
            .color = ctx.color("text_disabled"),
            .fontSize = m.font,
            .alignment = {0, 1},
        }},
        .clip = rects.input,
        .fontName = Typography::BodyFont,
        .maxCharacters = kMaxCharacters,
    });
    layoutChild(ctx, state.placeholder, rects.input);

    const Rect iconRect = {bounds.x + m.pad, rects.input.y, m.iconWidth, m.lineHeight};
    state.icon.update({
        .instances = {{
            .rect = iconRect,
            .str = style.icon,
            .visible = visible && !style.icon.empty(),
            .color = ctx.color(style.iconColorKey),
            .fontSize = m.font * kIconScale,
            .alignment = {1, 1},
        }},
        .clip = bounds,
        .fontName = Typography::IconFont,
        .maxCharacters = 4,
    });
    layoutChild(ctx, state.icon, iconRect);

    layoutChild(ctx, state.input, rects.input);

    const bool voice = state.voiceReady();
    state.send.update({
        .str = config.busy ? ICON_LC_SQUARE : voice ? "" : ICON_LC_ARROW_UP,
        .fontSize = style.fontSize,
        .rounding = 0.5f * state.controlSize(),
        .disabled = !config.busy && !voice && Jetstream::Text::IsBlank(state.text),
        .colorKey = "contrast_btn",
        .hoveredColorKey = "contrast_btn_hovered",
        .activeColorKey = "contrast_btn_active",
        .textColorKey = "contrast_btn_text",
        .onClick = [this]() {
            impl->press();
        },
    });
    layoutChild(ctx, state.send, rects.send);

    const F32 barWidth = std::round(m.control * kVoiceBarWidth);
    const F32 barGap = std::round(m.control * kVoiceBarGap);
    const F32 barsHeight = m.control * kVoiceIconHeight;
    const F32 barsWidth = static_cast<F32>(kVoiceBarHeights.size()) * barWidth +
                          static_cast<F32>(kVoiceBarHeights.size() - 1) * barGap;
    const F32 barsX = std::round(rects.send.x + (rects.send.width - barsWidth) * 0.5f);
    const F32 barsCenter = rects.send.y + rects.send.height * 0.5f;
    std::vector<Box::Instance> bars;
    for (U64 i = 0; i < kVoiceBarHeights.size(); ++i) {
        const F32 height = std::max(barWidth, std::round(barsHeight * kVoiceBarHeights[i]));
        bars.push_back({
            .rect = {barsX + static_cast<F32>(i) * (barWidth + barGap), std::round(barsCenter - height * 0.5f), barWidth,
                     height},
            .visible = visible && voice,
            .backgroundColor = ctx.color("contrast_btn_text"),
        });
    }
    state.voiceBars.update({
        .instances = std::move(bars),
        .clip = rects.send,
        .cornerRadius = barWidth * 0.5f,
        .capacity = kVoiceBarHeights.size(),
    });
    layoutChild(ctx, state.voiceBars, rects.send);

    if (state.showsControls()) {
        state.layoutControls(*this, ctx, m, outer, bounds, rects.send);
    } else {
        state.usageRect = {};
        state.layoutUsage(*this, ctx, m, outer);
    }
}

bool ChatComposer::mouseEvent(const MouseEvent& event) {
    auto& state = *impl;
    if (event.type == MouseEventType::Move || event.type == MouseEventType::Leave ||
        event.type == MouseEventType::Cancel) {
        const bool hovered = event.type == MouseEventType::Move && state.showsUsage() &&
                             state.usageRect.contains(event.position.x, event.position.y);
        if (hovered != state.usageHovered) {
            state.usageHovered = hovered;
            invalidate(Dirty::Layout);
        }
    }
    return mouseEventChildren(event);
}

}  // namespace Jetstream::Sakura
