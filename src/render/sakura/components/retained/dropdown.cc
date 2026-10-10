#include <jetstream/render/sakura/components/retained/dropdown.hh>

#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/label.hh>

#include "../../helpers.hh"
#include "../../retained/helpers.hh"
#include "../../retained/shapes.hh"
#include "../../retained/text_metrics.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kHorizontalPadding = 8.0f;
constexpr F32 kVerticalPadding = 4.0f;
constexpr F32 kCaretReserve = 14.0f;
constexpr F32 kPopupGap = 4.0f;
constexpr F32 kPopupPadding = 4.0f;
constexpr F32 kDetailGap = 5.0f;

}  // namespace

struct Dropdown::Impl {
    explicit Impl(Dropdown& self) : self(self) {}

    struct FieldColors {
        ColorRGBA<F32> background;
        ColorRGBA<F32> text;
        ColorRGBA<F32> border;
        ColorRGBA<F32> caret;
        ColorRGBA<F32> detail;
    };

    Dropdown& self;
    Config config;
    Box fieldBox;
    Label fieldLabel;
    Box caretBox;
    Box popupBox;
    Box rowBox;
    Label rowLabels;
    mutable TextMetrics textMetrics;
    Rect popupRect;
    F32 rowHeight = 0.0f;
    F32 popupInset = 0.0f;
    U64 poolRows = 1;
    U64 visibleRows = 0;
    U64 firstRow = 0;
    PressState press;
    bool open = false;
    I64 hoveredRow = -1;

    const Option* selected() const {
        const auto it = std::find_if(config.options.begin(), config.options.end(),
            [this](const auto& option) { return option.id == config.value; });
        return it == config.options.end() ? nullptr : &*it;
    }

    const std::string& label() const {
        const auto* option = selected();
        return option ? option->label : config.placeholder;
    }

    const std::string& detail() const {
        static const std::string empty;
        const auto* option = selected();
        return option ? option->detail : empty;
    }

    F32 textWidth(const std::string& label, const std::string& detail, F32 ratio) const {
        F32 width = textMetrics.measure(config.fontName, label, config.fontSize);
        if (!detail.empty()) {
            width += kDetailGap * ratio + textMetrics.measure(config.fontName, detail, config.fontSize);
        }
        return width;
    }

    F32 widestOption(F32 ratio) const {
        F32 width = 0.0f;
        for (const auto& option : config.options) {
            width = std::max(width, textWidth(option.label, option.detail, ratio));
        }
        return width;
    }

    I64 rowAt(F32 y) const {
        if (rowHeight <= 0.0f) {
            return -1;
        }
        const F32 top = popupRect.y + popupInset;
        if (y < top) {
            return -1;
        }
        const U64 slot = static_cast<U64>((y - top) / rowHeight);
        if (slot >= visibleRows || firstRow + slot >= config.options.size()) {
            return -1;
        }
        return static_cast<I64>(firstRow + slot);
    }

    FieldColors fieldColors(const Context& ctx) const {
        FieldColors colors = {
            .background = ctx.color(press.colorKey(config, open)),
            .text = ctx.color(config.textColorKey),
            .border = ctx.color(config.borderColorKey),
            .caret = ctx.color(config.caretColorKey),
            .detail = ctx.color(config.detailColorKey),
        };
        if (config.disabled) {
            colors.background.a *= config.disabledAlpha;
            colors.text.a *= config.disabledAlpha;
            colors.border.a *= config.disabledAlpha;
            colors.caret.a *= config.disabledAlpha;
            colors.detail.a *= config.disabledAlpha;
        }
        return colors;
    }

    void syncField(const FieldColors& colors, const Rect& bounds, const Rect& fieldClip,
                   const Rect& valueRect, F32 ratio, bool visible) {
        fieldBox.update({
            .id = config.id + ":bg",
            .instances = {{.rect = bounds, .visible = visible, .backgroundColor = colors.background}},
            .clip = fieldClip,
            .cornerRadius = config.cornerRadius,
            .borderWidth = config.borderWidth,
            .borderColor = colors.border,
        });

        const F32 labelWidth = textMetrics.measure(config.fontName, label(), config.fontSize);
        const F32 detailX = std::min(valueRect.right(), valueRect.x + labelWidth + kDetailGap * ratio);
        fieldLabel.update({
            .id = config.id + ":label",
            .instances = {
                {
                    .rect = valueRect,
                    .str = label(),
                    .visible = visible,
                    .color = colors.text,
                    .fontSize = config.fontSize,
                    .alignment = {0, 1},
                },
                {
                    .rect = {detailX, valueRect.y, std::max(0.0f, valueRect.right() - detailX), valueRect.height},
                    .str = detail(),
                    .visible = visible && !detail().empty(),
                    .color = colors.detail,
                    .fontSize = config.fontSize,
                    .alignment = {0, 1},
                },
            },
            .clip = Intersect(fieldClip, valueRect),
            .fontName = config.fontName,
            .sharpness = 0.45f,
            .maxCharacters = config.maxCharacters,
            .capacity = 2,
        });
    }

    void placePopup(const Context& ctx, const Rect& bounds, const Rect& area, F32 ratio, F32 hpad) {
        rowHeight = config.fontSize + 2.0f * kVerticalPadding * ratio;
        popupInset = kPopupPadding * ratio;
        poolRows = std::max<U64>(1, static_cast<U64>(static_cast<F32>(ctx.framebufferSize.y) /
                                                     std::max(1.0f, rowHeight)));
        const U64 rows = config.options.size();
        const F32 gap = kPopupGap * ratio;
        const F32 space = config.popupAbove ? bounds.y - gap - area.y : area.bottom() - bounds.bottom() - gap;
        const U64 fit = static_cast<U64>(std::max(0.0f, space - 2.0f * popupInset) / rowHeight);
        visibleRows = std::min({rows, poolRows, std::max<U64>(1, fit)});
        firstRow = std::min(firstRow, rows - visibleRows);
        const F32 popupHeight = visibleRows * rowHeight + 2.0f * popupInset;
        const F32 popupWidth = std::max(bounds.width, widestOption(ratio) + 2.0f * hpad);
        popupRect = {
            config.popupAlignRight ? bounds.right() - popupWidth : bounds.x,
            config.popupAbove ? bounds.y - gap - popupHeight : bounds.bottom() + gap,
            popupWidth,
            popupHeight,
        };
        if (hoveredRow >= 0 && static_cast<U64>(hoveredRow) >= rows) {
            hoveredRow = -1;
        }
    }

    void syncPopup(const Context& ctx, const Rect& popupClip, F32 ratio, bool popupVisible) {
        popupBox.update({
            .id = config.id + ":popup",
            .instances = {{
                .rect = popupRect,
                .visible = popupVisible,
                .backgroundColor = ctx.color(config.popupColorKey),
            }},
            .clip = popupClip,
            .cornerRadius = config.popupCornerRadius,
            .borderWidth = 1.0f * ratio,
            .borderColor = ctx.color(config.popupBorderColorKey),
        });

        const Rect hoveredRowRect = {
            popupRect.x + popupInset,
            popupRect.y + popupInset + (hoveredRow - static_cast<I64>(firstRow)) * rowHeight,
            std::max(0.0f, popupRect.width - 2.0f * popupInset),
            rowHeight,
        };
        rowBox.update({
            .id = config.id + ":row",
            .instances = {{
                .rect = hoveredRowRect,
                .visible = popupVisible && hoveredRow >= 0,
                .backgroundColor = ctx.color(config.rowHoveredColorKey),
            }},
            .clip = popupClip,
            .cornerRadius = std::max(0.0f, config.popupCornerRadius - popupInset),
        });
    }

    void syncRows(const Context& ctx, const Rect& popupClip, F32 ratio, F32 hpad, bool popupVisible) {
        std::vector<Label::Instance> rowInstances(2 * visibleRows);
        for (U64 i = 0; i < visibleRows; ++i) {
            const auto& option = config.options[firstRow + i];
            const Rect rowRect = {
                popupRect.x + hpad,
                popupRect.y + popupInset + i * rowHeight,
                std::max(0.0f, popupRect.width - 2.0f * hpad),
                rowHeight,
            };
            const F32 rowDetailX = rowRect.x + kDetailGap * ratio +
                textMetrics.measure(config.fontName, option.label, config.fontSize);
            rowInstances[2 * i] = {
                .rect = rowRect,
                .str = option.label,
                .visible = popupVisible,
                .color = ctx.color(option.id == config.value ? config.selectedTextColorKey : config.textColorKey),
                .fontSize = config.fontSize,
                .alignment = {0, 1},
            };
            rowInstances[2 * i + 1] = {
                .rect = {rowDetailX, rowRect.y, std::max(0.0f, rowRect.right() - rowDetailX), rowRect.height},
                .str = option.detail,
                .visible = popupVisible && !option.detail.empty(),
                .color = ctx.color(config.detailColorKey),
                .fontSize = config.fontSize,
                .alignment = {0, 1},
            };
        }

        rowLabels.update({
            .id = config.id + ":rows",
            .instances = std::move(rowInstances),
            .clip = popupClip,
            .fontName = config.fontName,
            .sharpness = 0.45f,
            .maxCharacters = config.maxCharacters,
            .capacity = 2 * poolRows,
        });
    }

    void setState(bool nextHovered, bool nextPressed, I64 nextRow) {
        const bool rowChanged = nextRow != hoveredRow;
        hoveredRow = nextRow;
        if (press.set(nextHovered, nextPressed) || rowChanged) {
            self.invalidate(Dirty::Paint);
        }
    }

    void setOpen(bool nextOpen) {
        if (nextOpen == open) {
            return;
        }
        open = nextOpen;
        firstRow = 0;
        self.invalidate(Dirty::Paint);
    }

    bool onMove(const MouseEvent& event, bool insideField, bool insidePopup) {
        if (insideField || insidePopup) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }
        setState(insideField, press.pressed && insideField, insidePopup ? rowAt(event.position.y) : -1);
        return insidePopup;
    }

    bool onScroll(const MouseEvent& event, bool insidePopup) {
        if (!insidePopup) {
            return false;
        }
        const U64 rows = config.options.size();
        const U64 last = rows - visibleRows;
        const U64 next = event.scroll.y > 0.0f ? (firstRow > 0 ? firstRow - 1 : 0)
                       : event.scroll.y < 0.0f ? std::min(last, firstRow + 1)
                                               : firstRow;
        if (next != firstRow) {
            firstRow = next;
            hoveredRow = rowAt(event.position.y);
            self.invalidate(Dirty::Paint);
        }
        return true;
    }

    bool onClick(bool insideField, bool insidePopup) {
        if (insideField) {
            setState(true, true, hoveredRow);
            return true;
        }
        if (insidePopup) {
            return true;
        }
        if (open) {
            setOpen(false);
            return true;
        }
        return false;
    }

    bool onRelease(const MouseEvent& event, bool insideField, bool insidePopup) {
        if (press.pressed) {
            setState(insideField, false, -1);
            if (insideField) {
                setOpen(!open);
            }
            return true;
        }
        if (!insidePopup) {
            return false;
        }
        const I64 row = rowAt(event.position.y);
        setOpen(false);
        if (row >= 0 && config.onSelect) {
            const auto selected = config.options[static_cast<U64>(row)].id;
            config.onSelect(selected);
        }
        return true;
    }
};

Dropdown::Dropdown() {
    this->impl = std::make_unique<Impl>(*this);
    add(this->impl->fieldBox);
    add(this->impl->fieldLabel);
    add(this->impl->caretBox);
    add(this->impl->popupBox);
    add(this->impl->rowBox);
    add(this->impl->rowLabels);
}

Dropdown::~Dropdown() = default;

bool Dropdown::update(Config config) {
    if (config.disabled || config.options.empty() ||
        config.id != impl->config.id || config.options != impl->config.options) {
        impl->setState(false, false, -1);
        impl->setOpen(false);
    }
    this->impl->config = std::move(config);
    return true;
}

Extent2D<F32> Dropdown::measure(const Context& ctx, Extent2D<F32> available) {
    impl->textMetrics.setWindow(ctx.render);

    const F32 ratio = ctx.pixelRatio;
    const F32 fontSizePixels = impl->config.fontSize;
    F32 textWidth = impl->textWidth(impl->label(), impl->detail(), ratio);
    if (!impl->config.fitValue) {
        textWidth = std::max(textWidth, impl->widestOption(ratio));
    }

    const F32 width = textWidth + (2.0f * kHorizontalPadding + kCaretReserve) * ratio;
    const F32 height = fontSizePixels + 2.0f * kVerticalPadding * ratio;

    return {std::min(width, available.x), std::min(height, available.y)};
}

void Dropdown::layout(const Context& ctx) {
    auto& state = *impl;
    const Rect bounds = frame();
    const Rect fieldClip = Intersect(frame(), clip());
    const Rect popupClip = clip();
    const bool visible = !bounds.empty();
    const F32 ratio = ctx.pixelRatio;
    const F32 hpad = kHorizontalPadding * ratio;
    const F32 caretWidth = kCaretReserve * ratio;
    const Rect valueRect = {
        bounds.x + hpad,
        bounds.y,
        std::max(0.0f, bounds.width - 2.0f * hpad - caretWidth),
        bounds.height,
    };
    const Rect caretRect = {
        bounds.right() - hpad - caretWidth,
        bounds.y,
        caretWidth,
        bounds.height,
    };

    const auto colors = state.fieldColors(ctx);
    state.syncField(colors, bounds, fieldClip, valueRect, ratio, visible);
    state.caretBox.update({
        .id = state.config.id + ":caret",
        .instances = {Caret(caretRect, true, {7.0f * ratio, 4.0f * ratio}, colors.caret, visible)},
        .clip = fieldClip,
        .shape = Box::Shape::Triangle,
        .capacity = 1,
    });

    state.placePopup(ctx, bounds, popupClip, ratio, hpad);
    const bool popupVisible = visible && state.open && !state.config.options.empty();
    state.syncPopup(ctx, popupClip, ratio, popupVisible);
    state.syncRows(ctx, popupClip, ratio, hpad, popupVisible);

    layoutChild(ctx, state.fieldBox, bounds);
    layoutChild(ctx, state.fieldLabel, bounds);
    layoutChild(ctx, state.caretBox, caretRect);
    layoutChild(ctx, state.popupBox, state.popupRect);
    layoutChild(ctx, state.rowBox, state.popupRect);
    layoutChild(ctx, state.rowLabels, state.popupRect);
}

bool Dropdown::event(const MouseEvent& event) {
    auto& state = *impl;
    if (state.config.disabled) {
        state.setState(false, false, -1);
        state.setOpen(false);
        return false;
    }

    const auto& point = event.position;
    const bool insideField = Intersect(frame(), clip()).contains(point.x, point.y);
    const bool insidePopup = state.open && Intersect(state.popupRect, clip()).contains(point.x, point.y);

    switch (event.type) {
        case MouseEventType::Move:
            return state.onMove(event, insideField, insidePopup);
        case MouseEventType::Scroll:
            return state.onScroll(event, insidePopup);
        case MouseEventType::Leave:
            state.setState(false, false, -1);
            return false;
        case MouseEventType::Click:
            return event.button == MouseButton::Left && state.onClick(insideField, insidePopup);
        case MouseEventType::Release:
            return event.button == MouseButton::Left && state.onRelease(event, insideField, insidePopup);
        default:
            return false;
    }
}

}  // namespace Jetstream::Sakura::Retained
