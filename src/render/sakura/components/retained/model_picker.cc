#include <jetstream/render/sakura/components/retained/model_picker.hh>

#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/label.hh>
#include <jetstream/render/tools/imgui_icons_ext.hh>

#include "../../helpers.hh"
#include "../../retained/helpers.hh"
#include "../../retained/shapes.hh"
#include "../../retained/text_metrics.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr U64 kMaxStops = 12;
constexpr F32 kThumbSpeed = 18.0f;

enum class Mode { Effort, Models };
enum class Hover { None, ModelRow, Row, Track };

}  // namespace

struct ModelPicker::Impl {
    explicit Impl(ModelPicker& self) : self(self) {}

    struct Metrics {
        F32 unit = 1.0f;
        F32 font = 0.0f;
        F32 hpad = 0.0f;
        F32 caret = 0.0f;
        F32 gap = 0.0f;
    };

    ModelPicker& self;
    Config config;
    Box fieldBox;
    Label fieldLabel;
    Box caretBox;
    Box popupBox;
    Box hoverBox;
    Box chevronBox;
    Box trackBox;
    Box fillBox;
    Box dotsBox;
    Box thumbBox;
    Label titleLabel;
    Label popupLabels;
    Label iconLabels;
    Label rowLabels;
    mutable TextMetrics textMetrics;
    PressState press;
    FrameClock clock;

    bool open = false;
    Mode mode = Mode::Effort;
    Hover hover = Hover::None;
    I64 hoveredRow = -1;
    bool dragging = false;
    I64 dragIndex = -1;
    std::string localEffort;
    std::string pendingModel;
    F32 thumbX = -1.0f;

    Rect bounds;
    Rect popupRect;
    Rect modelRowRect;
    Rect trackRect;
    F32 rowHeight = 0.0f;
    F32 listTop = 0.0f;
    U64 poolRows = 1;
    U64 visibleRows = 0;
    U64 firstRow = 0;
    std::vector<F32> stops;

    Metrics metrics() const {
        const F32 unit = config.fontSize / Typography::FontSize;
        return {unit, config.fontSize, 10.0f * unit, 12.0f * unit, 6.0f * unit};
    }

    const Option* selectedModel() const {
        const auto it = std::find_if(config.models.begin(), config.models.end(),
                                     [&](const auto& option) { return option.id == config.value; });
        return it == config.models.end() ? nullptr : &*it;
    }

    const std::string& currentEffort() const {
        return localEffort.empty() ? config.effort : localEffort;
    }

    I64 effortIndex() const {
        if (dragging) {
            return dragIndex;
        }
        for (U64 i = 0; i < config.efforts.size() && i < kMaxStops; ++i) {
            if (config.efforts[i].id == currentEffort()) {
                return static_cast<I64>(i);
            }
        }
        return -1;
    }

    const std::string& effortLabel() const {
        static const std::string empty;
        const I64 index = effortIndex();
        return index >= 0 ? config.efforts[static_cast<U64>(index)].label : empty;
    }

    F32 measureText(const std::string& text, F32 size) const {
        return text.empty() ? 0.0f : textMetrics.measure(config.fontName, text, size);
    }

    ColorRGBA<F32> color(const Context& ctx, const std::string& key) const {
        auto value = ctx.color(key);
        if (config.disabled) {
            value.a *= config.disabledAlpha;
        }
        return value;
    }

    I64 nearestStop(F32 x) const {
        if (stops.empty()) {
            return -1;
        }
        U64 best = 0;
        for (U64 i = 1; i < stops.size(); ++i) {
            if (std::abs(stops[i] - x) < std::abs(stops[best] - x)) {
                best = i;
            }
        }
        return static_cast<I64>(best);
    }

    I64 rowAt(F32 y) const {
        if (mode != Mode::Models || rowHeight <= 0.0f || y < listTop) {
            return -1;
        }
        const U64 slot = static_cast<U64>((y - listTop) / rowHeight);
        if (slot >= visibleRows || firstRow + slot >= config.models.size()) {
            return -1;
        }
        return static_cast<I64>(firstRow + slot);
    }

    void setOpen(bool next) {
        if (open == next) {
            return;
        }
        open = next;
        mode = config.efforts.empty() ? Mode::Models : Mode::Effort;
        hover = Hover::None;
        hoveredRow = -1;
        dragging = false;
        firstRow = 0;
        self.invalidate(Dirty::Paint);
    }

    void setHover(Hover next, I64 row) {
        if (hover != next || hoveredRow != row) {
            hover = next;
            hoveredRow = row;
            self.invalidate(Dirty::Paint);
        }
    }

    void commitEffort() {
        dragging = false;
        if (dragIndex < 0 || static_cast<U64>(dragIndex) >= config.efforts.size()) {
            return;
        }
        const auto& id = config.efforts[static_cast<U64>(dragIndex)].id;
        if (id != currentEffort()) {
            localEffort = id;
            if (config.onEffort) {
                config.onEffort(id);
            }
        }
        self.invalidate(Dirty::Paint);
    }

    void layoutField(const Context& ctx, const Metrics& m, const Rect& clip) {
        const auto* model = selectedModel();
        const std::string& label = model ? model->label : config.placeholder;
        const F32 labelWidth = measureText(label, m.font);
        const Rect textRect = {bounds.x + m.hpad, bounds.y,
                               std::max(0.0f, bounds.width - 2.0f * m.hpad - m.caret - m.gap * 0.5f), bounds.height};
        const F32 effortX = std::min(textRect.right(), textRect.x + labelWidth + m.gap);
        const bool visible = !bounds.empty();
        fieldBox.update({
            .id = config.id + ":field",
            .instances = {{.rect = bounds, .visible = visible, .backgroundColor = color(ctx, press.colorKey(config, open))}},
            .clip = clip,
            .cornerRadius = config.cornerRadius,
        });
        fieldLabel.update({
            .id = config.id + ":field-label",
            .instances = {
                {.rect = textRect, .str = label, .visible = visible, .color = color(ctx, config.textColorKey),
                 .fontSize = m.font, .alignment = {0, 1}},
                {.rect = {effortX, bounds.y, std::max(0.0f, textRect.right() - effortX), bounds.height},
                 .str = effortLabel(), .visible = visible && !effortLabel().empty(),
                 .color = color(ctx, config.detailColorKey), .fontSize = m.font, .alignment = {0, 1}},
            },
            .clip = Intersect(clip, textRect),
            .fontName = config.fontName,
            .maxCharacters = config.maxCharacters,
            .capacity = 2,
        });
        const Rect caretRect = {bounds.right() - m.hpad - m.caret, bounds.y + 1.5f * m.unit, m.caret, bounds.height};
        caretBox.update({
            .id = config.id + ":caret",
            .instances = {Caret(caretRect, true, {8.5f * m.unit, 5.0f * m.unit},
                                color(ctx, config.detailColorKey), visible)},
            .clip = clip,
            .shape = Box::Shape::Triangle,
            .capacity = 1,
        });
        self.layoutChild(ctx, fieldBox, bounds);
        self.layoutChild(ctx, fieldLabel, bounds);
        self.layoutChild(ctx, caretBox, caretRect);
    }

    void placePopup(const Rect& area, F32 width, F32 height, const Metrics& m) {
        const F32 x = config.popupAlignRight ? bounds.right() - width : bounds.x;
        const F32 offset = 3.0f * m.unit;
        const F32 y = config.popupAbove ? bounds.y - offset - height : bounds.bottom() + offset;
        popupRect = {std::round(std::max(area.x, x)), std::round(y), std::round(width), std::round(height)};
    }

    void layoutEffort(const Context& ctx, const Metrics& m, const Rect& area, bool shown) {
        const F32 pad = 10.0f * m.unit;
        const F32 titleHeight = m.font * 1.3f;
        const F32 modelHeight = m.font * 1.8f;
        const F32 thumbSize = 26.0f * m.unit;
        const F32 trackHeight = 20.0f * m.unit;
        const F32 width = std::max(bounds.width, 220.0f * m.unit);
        placePopup(area, width, pad + titleHeight + modelHeight + m.gap + thumbSize + pad, m);

        const Rect titleRect = {popupRect.x, popupRect.y + pad, width, titleHeight};
        modelRowRect = {popupRect.x + pad, titleRect.bottom(), width - 2.0f * pad, modelHeight};
        const F32 trackCenter = modelRowRect.bottom() + m.gap + thumbSize * 0.5f;
        trackRect = {popupRect.x + pad, trackCenter - trackHeight * 0.5f, width - 2.0f * pad, trackHeight};

        const U64 count = std::min<U64>(config.efforts.size(), kMaxStops);
        stops.clear();
        const F32 inset = trackHeight * 0.5f;
        for (U64 i = 0; i < count; ++i) {
            const F32 t = count == 1 ? 0.5f : static_cast<F32>(i) / static_cast<F32>(count - 1);
            stops.push_back(trackRect.x + inset + t * (trackRect.width - 2.0f * inset));
        }

        const I64 index = effortIndex();
        const F32 dt = clock.tick(std::chrono::steady_clock::now(), 0.1f);
        if (index >= 0) {
            const F32 target = stops[static_cast<U64>(index)];
            thumbX = thumbX < 0.0f ? target : Approach(thumbX, target, dt * kThumbSpeed);
            if (std::abs(thumbX - target) < 0.5f) {
                thumbX = target;
            }
        } else {
            thumbX = -1.0f;
        }

        const auto* model = selectedModel();
        const std::string modelLabel = model ? model->label : config.placeholder;
        const F32 modelWidth = measureText(modelLabel, m.font);
        const F32 chevron = 5.0f * m.unit;
        const F32 modelX = modelRowRect.x + (modelRowRect.width - modelWidth - m.gap - chevron) * 0.5f;
        const bool modelHovered = hover == Hover::ModelRow;
        const auto& title = effortLabel().empty() ? config.effortPlaceholder : effortLabel();

        titleLabel.update({
            .id = config.id + ":title",
            .instances = {{.rect = titleRect, .str = title, .visible = shown,
                           .color = color(ctx, effortLabel().empty() ? config.detailColorKey : config.accentColorKey),
                           .fontSize = m.font * 1.12f, .alignment = {1, 1}}},
            .clip = area,
            .fontName = config.fontName,
            .maxCharacters = config.maxCharacters,
            .capacity = 1,
        });
        popupLabels.update({
            .id = config.id + ":popup-labels",
            .instances = {{.rect = {modelX, modelRowRect.y, modelWidth + 1.0f, modelRowRect.height}, .str = modelLabel,
                           .visible = shown,
                           .color = color(ctx, modelHovered ? config.textColorKey : config.detailColorKey),
                           .fontSize = m.font, .alignment = {0, 1}}},
            .clip = area,
            .fontName = config.fontName,
            .maxCharacters = config.maxCharacters,
            .capacity = 1,
        });
        const Rect chevronRect = {modelX + modelWidth + m.gap, modelRowRect.y + 1.5f * m.unit, chevron,
                                  modelRowRect.height};
        chevronBox.update({
            .id = config.id + ":chevron",
            .instances = {Caret(chevronRect, false, {8.5f * m.unit, 5.0f * m.unit},
                                color(ctx, modelHovered ? config.textColorKey : config.detailColorKey), shown)},
            .clip = area,
            .shape = Box::Shape::Triangle,
            .capacity = 1,
        });
        iconLabels.update({
            .id = config.id + ":icons",
            .instances = {{.rect = chevronRect, .str = ICON_FA_CHECK, .visible = false}},
            .clip = area,
            .fontName = Typography::IconFont,
            .maxCharacters = 4,
            .capacity = 1,
        });
        hoverBox.update({
            .id = config.id + ":hover",
            .instances = {{.rect = modelRowRect, .visible = shown && modelHovered,
                           .backgroundColor = color(ctx, config.rowHoveredColorKey)}},
            .clip = area,
            .cornerRadius = modelRowRect.height * 0.5f,
        });

        trackBox.update({
            .id = config.id + ":track",
            .instances = {{.rect = trackRect, .visible = shown, .backgroundColor = color(ctx, config.trackColorKey)}},
            .clip = area,
            .cornerRadius = trackHeight * 0.5f,
        });
        const bool hasThumb = shown && thumbX >= 0.0f;
        fillBox.update({
            .id = config.id + ":fill",
            .instances = {{.rect = {trackRect.x, trackRect.y, hasThumb ? thumbX + inset - trackRect.x : 0.0f,
                                    trackRect.height},
                           .visible = hasThumb, .backgroundColor = color(ctx, config.accentColorKey)}},
            .clip = area,
            .cornerRadius = trackHeight * 0.5f,
        });
        const F32 dot = 4.0f * m.unit;
        std::vector<Box::Instance> dots;
        auto dotColor = color(ctx, config.detailColorKey);
        dotColor.a *= 0.8f;
        for (const F32 x : stops) {
            dots.push_back({.rect = {x - dot * 0.5f, trackRect.center().y - dot * 0.5f, dot, dot},
                            .visible = shown && (!hasThumb || x > thumbX + thumbSize * 0.5f),
                            .backgroundColor = dotColor});
        }
        dotsBox.update({
            .id = config.id + ":dots",
            .instances = std::move(dots),
            .clip = area,
            .cornerRadius = dot * 0.5f,
            .capacity = kMaxStops,
        });
        thumbBox.update({
            .id = config.id + ":thumb",
            .instances = {{.rect = {thumbX - thumbSize * 0.5f, trackCenter - thumbSize * 0.5f, thumbSize, thumbSize},
                           .visible = hasThumb, .backgroundColor = color(ctx, config.thumbColorKey)}},
            .clip = area,
            .cornerRadius = thumbSize * 0.5f,
        });
        rowLabels.update({
            .id = config.id + ":rows",
            .instances = {},
            .clip = area,
            .fontName = config.fontName,
            .maxCharacters = config.maxCharacters,
            .capacity = 2 * poolRows,
        });
    }

    void layoutModels(const Context& ctx, const Metrics& m, const Rect& area, bool shown) {
        const F32 pad = 6.0f * m.unit;
        const F32 inset = 12.0f * m.unit;
        rowHeight = m.font + 10.0f * m.unit;
        F32 widest = 0.0f;
        for (const auto& option : config.models) {
            widest = std::max(widest, measureText(option.label, m.font) + m.gap + measureText(option.detail, m.font));
        }
        const F32 width = std::max({bounds.width, 220.0f * m.unit, widest + 2.0f * inset + m.font * 2.0f});
        const U64 rows = config.models.size();
        const F32 space = config.popupAbove ? bounds.y - 3.0f * m.unit - area.y
                                            : area.bottom() - bounds.bottom() - 3.0f * m.unit;
        const U64 fit = static_cast<U64>(std::max(0.0f, space - 2.0f * pad) / rowHeight);
        visibleRows = std::min({rows, poolRows, std::max<U64>(1, fit)});
        firstRow = std::min(firstRow, rows - visibleRows);
        placePopup(area, width, visibleRows * rowHeight + 2.0f * pad, m);
        listTop = popupRect.y + pad;

        std::vector<Label::Instance> labels(2 * visibleRows);
        std::vector<Label::Instance> icons(1);
        for (U64 i = 0; i < visibleRows; ++i) {
            const auto& option = config.models[firstRow + i];
            const Rect row = {popupRect.x + inset, listTop + i * rowHeight, width - 2.0f * inset, rowHeight};
            const F32 labelWidth = measureText(option.label, m.font);
            const bool selected = option.id == config.value;
            labels[2 * i] = {.rect = row, .str = option.label, .visible = shown,
                             .color = color(ctx, config.textColorKey), .fontSize = m.font, .alignment = {0, 1}};
            labels[2 * i + 1] = {.rect = {row.x + labelWidth + m.gap, row.y,
                                          std::max(0.0f, row.width - labelWidth - m.gap - m.font * 1.5f), row.height},
                                 .str = option.detail, .visible = shown && !option.detail.empty(),
                                 .color = color(ctx, config.detailColorKey), .fontSize = m.font, .alignment = {0, 1}};
            if (selected) {
                icons[0] = {.rect = {row.right() - m.font, row.y, m.font, row.height}, .str = ICON_FA_CHECK,
                            .visible = shown, .color = color(ctx, config.textColorKey), .fontSize = m.font * 0.8f,
                            .alignment = {1, 1}};
            }
        }
        chevronBox.update({
            .id = config.id + ":chevron",
            .instances = {{.visible = false}},
            .clip = area,
            .shape = Box::Shape::Triangle,
            .capacity = 1,
        });
        rowLabels.update({
            .id = config.id + ":rows",
            .instances = std::move(labels),
            .clip = area,
            .fontName = config.fontName,
            .maxCharacters = config.maxCharacters,
            .capacity = 2 * poolRows,
        });
        iconLabels.update({
            .id = config.id + ":icons",
            .instances = std::move(icons),
            .clip = area,
            .fontName = Typography::IconFont,
            .maxCharacters = 4,
            .capacity = 1,
        });
        const bool rowHovered = shown && hoveredRow >= static_cast<I64>(firstRow) &&
                                hoveredRow < static_cast<I64>(firstRow + visibleRows);
        hoverBox.update({
            .id = config.id + ":hover",
            .instances = {{.rect = {popupRect.x + pad, listTop + (hoveredRow - static_cast<I64>(firstRow)) * rowHeight,
                                    width - 2.0f * pad, rowHeight},
                           .visible = rowHovered, .backgroundColor = color(ctx, config.rowHoveredColorKey)}},
            .clip = area,
            .cornerRadius = std::max(0.0f, config.popupCornerRadius - pad),
        });
        titleLabel.update({.id = config.id + ":title", .instances = {}, .clip = area,
                           .fontName = config.fontName, .maxCharacters = config.maxCharacters, .capacity = 1});
        popupLabels.update({.id = config.id + ":popup-labels", .instances = {}, .clip = area,
                            .fontName = config.fontName, .maxCharacters = config.maxCharacters, .capacity = 1});
        trackBox.update({.id = config.id + ":track", .instances = {}, .clip = area});
        fillBox.update({.id = config.id + ":fill", .instances = {}, .clip = area});
        thumbBox.update({.id = config.id + ":thumb", .instances = {}, .clip = area});
        dotsBox.update({.id = config.id + ":dots", .instances = {}, .clip = area, .capacity = kMaxStops});
    }
};

ModelPicker::ModelPicker() {
    impl = std::make_unique<Impl>(*this);
    add(impl->fieldBox);
    add(impl->fieldLabel);
    add(impl->caretBox);
    add(impl->popupBox);
    add(impl->hoverBox);
    add(impl->chevronBox);
    add(impl->trackBox);
    add(impl->fillBox);
    add(impl->dotsBox);
    add(impl->thumbBox);
    add(impl->titleLabel);
    add(impl->popupLabels);
    add(impl->iconLabels);
    add(impl->rowLabels);
}

ModelPicker::~ModelPicker() = default;

bool ModelPicker::update(Config config) {
    auto& state = *impl;
    const bool selected = !state.pendingModel.empty() && config.value == state.pendingModel;
    if (selected) {
        state.pendingModel.clear();
        state.thumbX = -1.0f;
        if (config.efforts.empty()) {
            state.setOpen(false);
        } else {
            state.mode = Mode::Effort;
            state.setHover(Hover::None, -1);
        }
    } else if (config.disabled || config.id != state.config.id || config.value != state.config.value ||
               config.efforts != state.config.efforts) {
        state.pendingModel.clear();
        state.setOpen(false);
        state.press.set(false, false);
    }
    if (config.effort != impl->config.effort || config.value != impl->config.value) {
        impl->localEffort.clear();
        invalidate(Dirty::Paint);
    }
    if (config.models != impl->config.models || config.efforts != impl->config.efforts) {
        invalidate(Dirty::Paint);
    }
    impl->config = std::move(config);
    return true;
}

Extent2D<F32> ModelPicker::measure(const Context& ctx, Extent2D<F32> available) {
    impl->textMetrics.setWindow(ctx.render);
    const auto m = impl->metrics();
    const auto* model = impl->selectedModel();
    F32 width = impl->measureText(model ? model->label : impl->config.placeholder, m.font);
    if (!impl->effortLabel().empty()) {
        width += m.gap + impl->measureText(impl->effortLabel(), m.font);
    }
    width += 2.0f * m.hpad + m.gap * 0.5f + m.caret;
    return {std::min(width, available.x), std::min(m.font + 8.0f * m.unit, available.y)};
}

void ModelPicker::layout(const Context& ctx) {
    auto& state = *impl;
    state.textMetrics.setWindow(ctx.render);
    state.bounds = frame();
    const auto m = state.metrics();
    const Rect area = clip();
    state.layoutField(ctx, m, Intersect(frame(), clip()));

    state.rowHeight = m.font + 10.0f * m.unit;
    state.poolRows = std::max<U64>(1, static_cast<U64>(static_cast<F32>(ctx.framebufferSize.y) /
                                                       std::max(1.0f, state.rowHeight)));
    if (state.open && state.mode == Mode::Effort && state.config.efforts.empty()) {
        state.mode = Mode::Models;
    }
    const bool visible = state.open && !state.bounds.empty();
    if (state.mode == Mode::Effort) {
        state.layoutEffort(ctx, m, area, visible);
    } else {
        state.layoutModels(ctx, m, area, visible);
    }

    if (!visible) {
        state.popupRect = {};
    }
    const auto popupColor = state.color(ctx, state.config.popupColorKey);
    state.popupBox.update({
        .id = state.config.id + ":popup",
        .instances = {{.rect = state.popupRect, .visible = visible, .backgroundColor = popupColor}},
        .clip = area,
        .cornerRadius = state.config.popupCornerRadius,
        .borderWidth = std::max(1.0f, std::round(m.unit)),
        .borderColor = Over(popupColor, state.color(ctx, state.config.popupBorderColorKey), popupColor.a),
    });
    for (Component* child : std::initializer_list<Component*>{
             &state.popupBox, &state.hoverBox, &state.chevronBox, &state.trackBox, &state.fillBox, &state.dotsBox,
             &state.thumbBox, &state.titleLabel, &state.popupLabels, &state.iconLabels, &state.rowLabels}) {
        layoutChild(ctx, *child, visible ? state.popupRect : Rect{});
    }
}

bool ModelPicker::event(const MouseEvent& event) {
    auto& state = *impl;
    if (state.config.disabled) {
        state.press.set(false, false);
        state.setOpen(false);
        return false;
    }
    const auto& point = event.position;
    const bool insideField = Intersect(frame(), clip()).contains(point.x, point.y);
    const bool insidePopup = state.open && Intersect(state.popupRect, clip()).contains(point.x, point.y);
    const bool effortMode = state.mode == Mode::Effort;
    const bool onTrack = effortMode && insidePopup && point.y >= state.modelRowRect.bottom();
    const bool onModelRow = effortMode && insidePopup && state.modelRowRect.contains(point.x, point.y);

    switch (event.type) {
        case MouseEventType::Move:
            if (state.dragging) {
                state.dragIndex = state.nearestStop(point.x);
                invalidate(Dirty::Paint);
                return true;
            }
            if (insideField || onTrack || onModelRow || state.rowAt(point.y) >= 0) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            }
            if (state.press.set(insideField, state.press.pressed && insideField)) {
                invalidate(Dirty::Paint);
            }
            state.setHover(onModelRow ? Hover::ModelRow : onTrack ? Hover::Track
                           : insidePopup && !effortMode ? Hover::Row : Hover::None,
                           insidePopup && !effortMode ? state.rowAt(point.y) : -1);
            return insidePopup;
        case MouseEventType::Scroll:
            if (insidePopup && !effortMode) {
                const U64 last = state.config.models.size() - state.visibleRows;
                const U64 next = event.scroll.y > 0.0f ? (state.firstRow > 0 ? state.firstRow - 1 : 0)
                               : event.scroll.y < 0.0f ? std::min(last, state.firstRow + 1) : state.firstRow;
                if (next != state.firstRow) {
                    state.firstRow = next;
                    state.hoveredRow = state.rowAt(point.y);
                    invalidate(Dirty::Paint);
                }
            }
            return insidePopup;
        case MouseEventType::Leave:
            if (!state.dragging) {
                state.press.set(false, false);
                state.setHover(Hover::None, -1);
            }
            return false;
        case MouseEventType::Click:
            if (event.button != MouseButton::Left) {
                return false;
            }
            if (insideField) {
                state.press.set(true, true);
                invalidate(Dirty::Paint);
                return true;
            }
            if (onTrack && !state.stops.empty()) {
                state.dragging = true;
                state.dragIndex = state.nearestStop(point.x);
                invalidate(Dirty::Paint);
                return true;
            }
            if (insidePopup) {
                return true;
            }
            if (state.open) {
                state.setOpen(false);
                return true;
            }
            return false;
        case MouseEventType::Release:
            if (event.button != MouseButton::Left) {
                return false;
            }
            if (state.dragging) {
                state.dragIndex = state.nearestStop(point.x);
                state.commitEffort();
                return true;
            }
            if (state.press.pressed) {
                state.press.set(insideField, false);
                if (insideField) {
                    state.setOpen(!state.open);
                }
                return true;
            }
            if (onModelRow) {
                state.mode = Mode::Models;
                state.setHover(Hover::None, -1);
                invalidate(Dirty::Paint);
                return true;
            }
            if (insidePopup && !effortMode) {
                const I64 row = state.rowAt(point.y);
                if (row >= 0) {
                    const auto id = state.config.models[static_cast<U64>(row)].id;
                    if (id == state.config.value) {
                        if (state.config.efforts.empty()) {
                            state.setOpen(false);
                        } else {
                            state.mode = Mode::Effort;
                            state.setHover(Hover::None, -1);
                        }
                    } else {
                        state.pendingModel = id;
                        if (state.config.onSelect) {
                            state.config.onSelect(id);
                        }
                    }
                }
                return true;
            }
            return false;
        default:
            return false;
    }
}

}  // namespace Jetstream::Sakura::Retained
