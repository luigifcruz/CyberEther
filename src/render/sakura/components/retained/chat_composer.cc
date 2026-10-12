#include <jetstream/render/sakura/components/retained/chat_composer.hh>

#include <jetstream/logger.hh>
#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/button.hh>
#include <jetstream/render/sakura/components/retained/label.hh>
#include <jetstream/render/sakura/components/retained/text_grid.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/imgui_icons_ext.hh>

#include "../../context.hh"
#include "../../state.hh"
#include "../../retained/helpers.hh"
#include "tools/text.hh"
#include "../../retained/text_metrics.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace Jetstream::Sakura::Retained {

using Jetstream::Text::IsBlank;

namespace {

constexpr std::array<F32, 5> kVoiceBarHeights = {0.45f, 1.0f, 0.6f, 0.85f, 0.3f};
constexpr F32 kVoiceIconHeightRatio = 0.44f;
constexpr F32 kVoiceBarWidthRatio = 0.075f;
constexpr F32 kVoiceBarGapRatio = 0.06f;
constexpr U64 kUsageRingDots = 48;
constexpr F32 kUsageRingRadiusRatio = 0.42f;
constexpr F32 kUsageRingDotRatio = 0.13f;
constexpr F32 kUsageRingTrackAlpha = 0.35f;
constexpr F32 kTooltipPaddingX = 14.0f;
constexpr F32 kTooltipPaddingY = 8.0f;
constexpr F32 kTooltipGap = 6.0f;
constexpr F32 kTooltipRadius = 14.0f;
constexpr U64 kTooltipLines = 3;

template<typename Config>
Config Transparent(Config config) {
    config.colorKey = "transparent";
    config.hoveredColorKey = "button_hovered";
    config.activeColorKey = "button_active";
    config.borderColorKey = "transparent";
    return config;
}

bool SameSelector(const std::optional<ChatComposer::Selector>& a,
                  const std::optional<ChatComposer::Selector>& b) {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    return !a || (a->options == b->options && a->value == b->value &&
                  a->placeholder == b->placeholder && a->disabled == b->disabled &&
                  a->efforts == b->efforts && a->effort == b->effort);
}

}  // namespace

struct ChatComposer::Impl {
    explicit Impl(ChatComposer& self) : self(self) {}

    ChatComposer& self;
    Config config;
    Box background;
    Label placeholder;
    Label icon;
    TextGrid input;
    Button sendButton;
    Box voiceBars;
    Button attachButton;
    Button clearButton;
    ModelPicker modelPicker;
    Box usageRing;
    Box tooltipBox;
    Label tooltipText;
    mutable TextMetrics textMetrics;

    Rect usageRect;
    bool usageHovered = false;
    std::string inputText;
    U64 focusRequest = 0;
    U64 clearRequest = 0;

    struct Metrics {
        F32 unit;
        F32 font;
        F32 lineHeight;
        F32 pad;
        F32 textPad;
        F32 controlsHeight;
        F32 sendSize;
        F32 iconWidth;
    };

    struct InputRects {
        Rect input;
        Rect send;
        F32 inputY;
    };

    Metrics metricsFor() const {
        const F32 font = config.style.fontSize;
        const F32 unit = font / Typography::FontSize;
        return {
            .unit = unit,
            .font = font,
            .lineHeight = font * Typography::CodeLineHeight,
            .pad = config.style.padding,
            .textPad = 6.0f * unit,
            .controlsHeight = font + 14.0f * unit,
            .sendSize = font + 14.0f * unit,
            .iconWidth = config.style.icon.empty() ? 0.0f : font * 1.8f,
        };
    }

    F32 inputWidthFor(const Metrics& m, F32 boxWidth) const {
        F32 width = std::max(0.0f, boxWidth - 2.0f * m.pad - m.iconWidth);
        if (config.style.compact) {
            width = std::max(0.0f, width - m.sendSize - m.textPad);
        }
        return width;
    }

    F32 textHeightFor(const Metrics& m, F32 measuredHeight) const {
        return measuredHeight + 2.0f * m.textPad;
    }

    bool voiceReady() const {
        return !config.busy && config.onVoice && IsBlank(inputText);
    }

    F32 usageValue() const {
        return config.usage ? std::clamp(config.usage->value, 0.0f, 1.0f) : 0.0f;
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

    void submit() {
        if (config.busy || IsBlank(inputText)) {
            return;
        }
        if (config.onSubmit) {
            config.onSubmit(inputText);
        }
    }

    void press() {
        if (config.busy) {
            if (config.onCancel) {
                config.onCancel();
            }
            return;
        }
        if (config.onVoice && IsBlank(inputText)) {
            config.onVoice();
            return;
        }
        submit();
    }

    void syncInput(const Metrics& m) {
        input.update({
            .id = jst::fmt::format("{}:input", config.id),
            .value = inputText,
            .editable = true,
            .fontSize = m.font,
            .fontName = "default_body",
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
            .onChange = [this](std::string text) {
                inputText = std::move(text);
            },
            .onSubmit = [this](std::string text) {
                inputText = std::move(text);
                submit();
            },
        });
    }

    Rect insetBounds(const Rect& outer) const {
        const auto& margin = config.style.margin;
        return {
            outer.x + margin.left,
            outer.y + margin.top,
            std::max(0.0f, outer.width - (margin.left + margin.right)),
            std::max(0.0f, outer.height - (margin.top + margin.bottom)),
        };
    }

    InputRects inputRectsFor(const Metrics& m, const Rect& bounds, F32 textHeight) const {
        const F32 rowHeight = bounds.height - 2.0f * m.pad;
        const F32 centerOffset = config.style.compact
            ? std::max(0.0f, (rowHeight - textHeight) * 0.5f) : 0.0f;
        const F32 inputX = bounds.x + m.pad + m.iconWidth;
        const F32 inputY = bounds.y + m.pad + centerOffset + m.textPad;
        if (config.style.compact) {
            const Rect send = {
                bounds.right() - m.pad - m.sendSize,
                bounds.bottom() - m.pad - m.sendSize,
                m.sendSize,
                m.sendSize,
            };
            const Rect input = {
                inputX,
                inputY,
                std::max(0.0f, send.x - m.textPad - inputX),
                std::max(0.0f, bounds.bottom() - m.pad - centerOffset - m.textPad - inputY),
            };
            return {input, send, inputY};
        }
        const Rect input = {
            inputX,
            inputY,
            std::max(0.0f, bounds.width - 2.0f * m.pad - m.iconWidth),
            std::max(0.0f, bounds.bottom() - m.pad - m.sendSize - m.textPad - inputY),
        };
        const Rect send = {
            bounds.right() - m.pad - m.sendSize,
            input.bottom() + m.textPad,
            m.sendSize,
            m.sendSize,
        };
        return {input, send, inputY};
    }

    void layoutBackground(const Context& ctx, const Metrics& m, const Rect& bounds, bool visible) {
        const auto card = ctx.color(config.style.backgroundColorKey);
        const auto border = Over(card, ctx.color(config.style.borderColorKey), card.a);
        background.update({
            .id = jst::fmt::format("{}:bg", config.id),
            .instances = {{.rect = bounds, .visible = visible, .backgroundColor = card}},
            .cornerRadius = config.style.cornerRadius.value_or(m.pad + m.sendSize * 0.5f),
            .borderWidth = config.style.borderWidth,
            .borderColor = border,
        });
        self.layoutChild(ctx, background, bounds);
    }

    InputRects layoutInput(const Context& ctx, const Metrics& m, const Rect& bounds) {
        syncInput(m);
        if (Private::ConsumeRequest(focusRequest, config.focusRequest)) {
            input.focus();
        }
        const F32 measuredHeight = self.measureChild(
            input, ctx, {inputWidthFor(m, bounds.width), std::numeric_limits<F32>::infinity()}).y;
        const auto rects = inputRectsFor(m, bounds, textHeightFor(m, measuredHeight));
        self.layoutChild(ctx, input, rects.input);
        return rects;
    }

    void layoutIcon(const Context& ctx, const Metrics& m, const Rect& bounds, F32 inputY, bool visible) {
        const Rect iconRect = {bounds.x + m.pad, inputY, m.iconWidth, m.lineHeight};
        icon.update({
            .id = jst::fmt::format("{}:icon", config.id),
            .instances = {{
                .rect = iconRect,
                .str = config.style.icon,
                .visible = visible && !config.style.icon.empty(),
                .color = ctx.color(config.style.iconColorKey),
                .fontSize = m.font * 1.1f,
                .alignment = {1, 1},
            }},
            .fontName = Typography::IconFont,
            .maxCharacters = 4,
        });
        self.layoutChild(ctx, icon, iconRect);
    }

    void layoutPlaceholder(const Context& ctx, const Metrics& m, const Rect& inputRect, bool visible) {
        const bool showPlaceholder = !config.style.placeholder.empty() && inputText.empty();
        placeholder.update({
            .id = jst::fmt::format("{}:placeholder", config.id),
            .instances = {{
                .rect = {
                    inputRect.x + m.textPad,
                    inputRect.y,
                    std::max(0.0f, inputRect.width - 2.0f * m.textPad),
                    m.lineHeight,
                },
                .str = config.style.placeholder,
                .visible = visible && showPlaceholder,
                .color = ctx.color("text_disabled"),
                .fontSize = m.font,
                .alignment = {0, 1},
            }},
            .clip = inputRect,
            .fontName = "default_body",
            .maxCharacters = std::max<U64>(128, config.style.placeholder.size()),
        });
        self.layoutChild(ctx, placeholder, inputRect);
    }

    void layoutSend(const Context& ctx, const Metrics& m, const Rect& sendRect) {
        const bool busy = config.busy;
        const bool voice = voiceReady();
        sendButton.update({
            .id = jst::fmt::format("{}:send", config.id),
            .str = busy ? ICON_FA_STOP : voice ? "" : ICON_FA_ARROW_UP,
            .disabled = !busy && !voice && IsBlank(inputText),
            .colorKey = "contrast_btn",
            .hoveredColorKey = "contrast_btn_hovered",
            .activeColorKey = "contrast_btn_active",
            .borderColorKey = "transparent",
            .textColorKey = "contrast_btn_text",
            .fontSize = m.font,
            .fontName = Typography::IconFont,
            .cornerRadius = m.sendSize * 0.5f,
            .onClick = [this] { press(); },
        });
        self.layoutChild(ctx, sendButton, sendRect);
    }

    void layoutVoiceBars(const Context& ctx, const Metrics& m, const Rect& sendRect, bool visible) {
        const F32 barWidth = std::max(1.0f, std::round(m.sendSize * kVoiceBarWidthRatio));
        const F32 barGap = std::max(1.0f, std::round(m.sendSize * kVoiceBarGapRatio));
        const F32 barsHeight = m.sendSize * kVoiceIconHeightRatio;
        const F32 barsWidth = kVoiceBarHeights.size() * barWidth +
                              (kVoiceBarHeights.size() - 1) * barGap;
        const F32 barsX = std::round(sendRect.x + (sendRect.width - barsWidth) * 0.5f);
        const F32 barsCenterY = sendRect.y + sendRect.height * 0.5f;
        std::vector<Box::Instance> bars;
        for (U64 i = 0; i < kVoiceBarHeights.size(); ++i) {
            const F32 height = std::max(barWidth, std::round(barsHeight * kVoiceBarHeights[i]));
            bars.push_back({
                .rect = {barsX + static_cast<F32>(i) * (barWidth + barGap),
                         std::round(barsCenterY - height * 0.5f), barWidth, height},
                .visible = visible,
                .backgroundColor = ctx.color("contrast_btn_text"),
            });
        }
        voiceBars.update({
            .id = jst::fmt::format("{}:voice-bars", config.id),
            .instances = std::move(bars),
            .clip = sendRect,
            .cornerRadius = barWidth * 0.5f,
            .capacity = kVoiceBarHeights.size(),
        });
        self.layoutChild(ctx, voiceBars, sendRect);
    }

    void layoutControls(const Context& ctx, const Metrics& m, const Rect& outer, const Rect& bounds,
                        const Rect& sendRect, bool visible) {
        const bool shown = visible && !config.style.compact;
        const F32 controlsY = sendRect.y + (m.sendSize - m.controlsHeight) * 0.5f;
        const Rect clearRect = layoutAttachClear(ctx, m, bounds, controlsY, shown);
        const Rect dropdownRect = layoutModelPicker(ctx, m, sendRect, clearRect, controlsY, shown);
        const bool showUsage = shown && config.usage.has_value();
        const Rect ringRect = showUsage
            ? Rect{dropdownRect.x - m.controlsHeight, controlsY, m.controlsHeight, m.controlsHeight}
            : Rect{};
        usageRect = ringRect;
        if (!showUsage) {
            usageHovered = false;
        }
        layoutUsageRing(ctx, m, ringRect, showUsage);
        layoutTooltip(ctx, m, outer, ringRect, showUsage && usageHovered);
    }

    Rect layoutAttachClear(const Context& ctx, const Metrics& m, const Rect& bounds, F32 controlsY, bool shown) {
        attachButton.update(Transparent<Button::Config>({
            .id = jst::fmt::format("{}:attach", config.id),
            .str = ICON_FA_PLUS,
            .textColorKey = "text_primary",
            .fontSize = m.font * 0.9f,
            .fontName = Typography::IconFont,
            .cornerRadius = m.controlsHeight * 0.5f,
            .onClick = [this] {
                if (config.onAttach) {
                    config.onAttach();
                }
            },
        }));
        const Rect attachRect = {bounds.x + m.pad, controlsY, m.controlsHeight, m.controlsHeight};
        self.layoutChild(ctx, attachButton, shown ? attachRect : Rect{});

        clearButton.update(Transparent<Button::Config>({
            .id = jst::fmt::format("{}:clear", config.id),
            .str = "Clear",
            .textColorKey = "text_secondary",
            .fontSize = m.font,
            .fontName = "default_body",
            .cornerRadius = m.controlsHeight * 0.5f,
            .horizontalPadding = m.controlsHeight * 0.5f,
            .onClick = [this] {
                if (config.onClear) {
                    config.onClear();
                }
            },
        }));
        const bool clear = shown && config.style.clearButton && config.onClear;
        const F32 clearWidth = clear
            ? self.measureChild(clearButton, ctx, {bounds.width, m.controlsHeight}).x
            : 0.0f;
        const Rect clearRect = {attachRect.right(), controlsY, clearWidth, m.controlsHeight};
        self.layoutChild(ctx, clearButton, clear ? clearRect : Rect{});
        return clearRect;
    }

    Rect layoutModelPicker(const Context& ctx, const Metrics& m, const Rect& sendRect, const Rect& clearRect,
                           F32 controlsY, bool shown) {
        const bool show = shown && config.selector.has_value();
        const Selector selector = config.selector.value_or(Selector{});
        modelPicker.update({
            .id = jst::fmt::format("{}:selector", config.id),
            .models = selector.options,
            .value = selector.value,
            .placeholder = selector.placeholder,
            .efforts = selector.efforts,
            .effort = selector.effort,
            .disabled = selector.disabled || !show,
            .accentColorKey = "agent_activity",
            .popupColorKey = config.style.backgroundColorKey,
            .fontSize = m.font,
            .cornerRadius = m.controlsHeight * 0.5f,
            .popupCornerRadius = 15.0f * m.unit,
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
        const F32 controlsGap = m.textPad;
        const F32 ringSize = config.usage ? m.controlsHeight : 0.0f;
        const F32 dropdownRight = sendRect.x - controlsGap;
        const F32 dropdownLimit = std::max(
            0.0f, dropdownRight - (clearRect.right() + controlsGap) - ringSize);
        const F32 dropdownWidth = show ? std::min(dropdownLimit,
            self.measureChild(modelPicker, ctx, {dropdownLimit, m.controlsHeight}).x) : 0.0f;
        const Rect dropdownRect = {dropdownRight - dropdownWidth, controlsY,
                                   dropdownWidth, m.controlsHeight};
        self.layoutChild(ctx, modelPicker, show ? dropdownRect : Rect{});
        return dropdownRect;
    }

    void layoutUsageRing(const Context& ctx, const Metrics& m, const Rect& ringRect, bool visible) {
        const F32 used = usageValue();
        const F32 ringRadius = m.font * kUsageRingRadiusRatio;
        const F32 dotSize = std::max(1.0f, m.font * kUsageRingDotRatio);
        const F32 centerX = ringRect.x + ringRect.width * 0.5f;
        const F32 centerY = ringRect.y + ringRect.height * 0.5f;
        const auto fill = ctx.color("text_primary");
        auto track = ctx.color("text_secondary");
        track.a *= kUsageRingTrackAlpha;
        std::vector<Box::Instance> dots;
        for (U64 i = 0; i < kUsageRingDots; ++i) {
            const F32 position = static_cast<F32>(i) / static_cast<F32>(kUsageRingDots);
            const F32 angle = (position * 2.0f - 0.5f) * static_cast<F32>(JST_PI);
            dots.push_back({
                .rect = {centerX + ringRadius * std::cos(angle) - dotSize * 0.5f,
                         centerY + ringRadius * std::sin(angle) - dotSize * 0.5f, dotSize, dotSize},
                .visible = visible,
                .backgroundColor = position < used ? fill : track,
            });
        }
        usageRing.update({
            .id = jst::fmt::format("{}:usage-ring", config.id),
            .instances = std::move(dots),
            .clip = ringRect,
            .cornerRadius = dotSize * 0.5f,
            .capacity = kUsageRingDots,
        });
        self.layoutChild(ctx, usageRing, ringRect);
    }

    void layoutTooltip(const Context& ctx, const Metrics& m, const Rect& outer,
                       const Rect& ringRect, bool visible) {
        const auto lines = tooltipLines();
        textMetrics.setWindow(ctx.render);
        F32 textWidth = 0.0f;
        for (const auto& line : lines) {
            textWidth = std::max(textWidth, textMetrics.measure("default_body", line, m.font));
        }
        const F32 tooltipLine = m.lineHeight * 1.15f;
        const F32 tooltipWidth = textWidth + 2.0f * kTooltipPaddingX * m.unit;
        const F32 tooltipHeight = lines.size() * tooltipLine + 2.0f * kTooltipPaddingY * m.unit;
        const F32 tooltipX = std::clamp(ringRect.x + ringRect.width * 0.5f - tooltipWidth * 0.5f,
                                        outer.x, std::max(outer.x, outer.right() - tooltipWidth));
        const Rect tooltipRect = {
            tooltipX,
            ringRect.y - kTooltipGap * m.unit - tooltipHeight,
            tooltipWidth,
            tooltipHeight,
        };
        tooltipBox.update({
            .id = jst::fmt::format("{}:tooltip", config.id),
            .instances = {{.rect = tooltipRect, .visible = visible,
                           .backgroundColor = ctx.color("popup_bg")}},
            .cornerRadius = kTooltipRadius * m.unit,
            .borderWidth = 1.0f * m.unit,
            .borderColor = ctx.color("border"),
        });
        self.layoutChild(ctx, tooltipBox, tooltipRect);

        std::vector<Label::Instance> tooltipLines;
        for (U64 i = 0; i < lines.size(); ++i) {
            tooltipLines.push_back({
                .rect = {tooltipRect.x, tooltipRect.y + kTooltipPaddingY * m.unit + i * tooltipLine,
                         tooltipRect.width, tooltipLine},
                .str = lines[i],
                .visible = visible,
                .color = ctx.color(i == 0 ? "text_secondary" : "text_primary"),
                .fontSize = m.font,
                .alignment = {1, 1},
            });
        }
        tooltipText.update({
            .id = jst::fmt::format("{}:tooltip-text", config.id),
            .instances = std::move(tooltipLines),
            .fontName = "default_body",
            .capacity = kTooltipLines,
        });
        self.layoutChild(ctx, tooltipText, tooltipRect);
    }
};

ChatComposer::ChatComposer() {
    impl = std::make_unique<Impl>(*this);
    add(impl->background);
    add(impl->placeholder);
    add(impl->icon);
    add(impl->input);
    add(impl->sendButton);
    add(impl->voiceBars);
    add(impl->attachButton);
    add(impl->clearButton);
    add(impl->modelPicker);
    add(impl->usageRing);
    add(impl->tooltipBox);
    add(impl->tooltipText);
}

ChatComposer::~ChatComposer() = default;

bool ChatComposer::update(Config config) {
    auto& current = impl->config;
    if (current.id != config.id || current.style != config.style || current.busy != config.busy ||
        !SameSelector(current.selector, config.selector) || current.usage != config.usage) {
        invalidate(Dirty::Paint);
    }
    if (Private::ConsumeRequest(impl->clearRequest, config.clearRequest) && !impl->inputText.empty()) {
        impl->inputText.clear();
        invalidate(Dirty::Paint);
    }
    current = std::move(config);
    return true;
}

Extent2D<F32> ChatComposer::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const auto m = impl->metricsFor();
    const F32 width = std::isfinite(available.x) ? available.x : 0.0f;
    const F32 boxWidth = std::max(0.0f, width - (config.style.margin.left + config.style.margin.right));
    impl->syncInput(m);
    const F32 measuredHeight = measureChild(
        impl->input, ctx,
        {impl->inputWidthFor(m, boxWidth), std::numeric_limits<F32>::infinity()}).y;
    const F32 textHeight = impl->textHeightFor(m, measuredHeight);
    const F32 boxHeight = config.style.compact ? std::max(textHeight, m.sendSize) + 2.0f * m.pad
                                         : textHeight + m.sendSize + 2.0f * m.pad;
    return {width, boxHeight + config.style.margin.top + config.style.margin.bottom};
}

void ChatComposer::layout(const Context& ctx) {
    auto& state = *impl;
    const auto m = state.metricsFor();
    const Rect outer = frame();
    const Rect bounds = state.insetBounds(outer);
    const bool visible = !bounds.empty();

    state.layoutBackground(ctx, m, bounds, visible);
    const auto rects = state.layoutInput(ctx, m, bounds);
    state.layoutIcon(ctx, m, bounds, rects.inputY, visible);
    state.layoutPlaceholder(ctx, m, rects.input, visible);
    state.layoutSend(ctx, m, rects.send);
    state.layoutVoiceBars(ctx, m, rects.send, visible && state.voiceReady());

    state.layoutControls(ctx, m, outer, bounds, rects.send, visible);
}

bool ChatComposer::event(const MouseEvent& event) {
    if (event.type == MouseEventType::Move || event.type == MouseEventType::Leave) {
        const bool hovered = event.type == MouseEventType::Move &&
                             impl->usageRect.contains(event.position.x, event.position.y);
        if (hovered != impl->usageHovered) {
            impl->usageHovered = hovered;
            invalidate(Dirty::Paint);
        }
    }
    return eventChildren(event);
}

}  // namespace Jetstream::Sakura::Retained
