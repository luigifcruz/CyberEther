#include <jetstream/render/sakura/components/model_picker.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/popup.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/popup_layer.hh"
#include "../detail/text_metrics.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr U64 kMaxStops = 12;
constexpr U64 kMaxCharacters = 128;
constexpr F32 kThumbSpeed = 18.0f;
constexpr F32 kThumbSnap = 0.5f;
constexpr F32 kFieldPaddingX = 10.0f;
constexpr F32 kFieldPaddingY = 4.0f;
constexpr F32 kCaretWidth = 12.0f;
constexpr F32 kCaretLength = 8.5f;
constexpr F32 kCaretDepth = 5.0f;
constexpr F32 kCaretDrop = 1.5f;
constexpr F32 kGap = 6.0f;
constexpr F32 kPanelWidth = 220.0f;
constexpr F32 kEffortPadding = 10.0f;
constexpr F32 kTitleHeight = 1.3f;
constexpr F32 kTitleScale = 1.12f;
constexpr F32 kModelRowHeight = 1.8f;
constexpr F32 kChevronScale = 0.6f;
constexpr F32 kThumbSize = 26.0f;
constexpr F32 kTrackHeight = 20.0f;
constexpr F32 kStopSize = 4.0f;
constexpr F32 kStopAlpha = 0.8f;
constexpr F32 kListPadding = 6.0f;
constexpr F32 kListInset = 12.0f;
constexpr F32 kRowPadding = 10.0f;
constexpr F32 kCheckScale = 0.8f;
constexpr F32 kCheckSpace = 1.5f;
constexpr F32 kMarkSpace = 2.0f;

enum class Mode {
    Effort,
    Models,
};

struct Target {
    enum class Kind {
        None,
        ModelRow,
        Row,
    };

    Kind kind = Kind::None;
    U64 row = 0;

    bool operator==(const Target&) const = default;
};

F32 StopFraction(U64 index, U64 count) {
    return count <= 1 ? 0.5f : static_cast<F32>(index) / static_cast<F32>(count - 1);
}

struct PickerMetrics {
    F32 unit = 1.0f;
    F32 font = 0.0f;
    F32 gap = 0.0f;
};

struct PickerState {
    ModelPicker::Config config;
    TextMetrics textMetrics;
    bool hovered = false;
    bool open = false;
    Mode mode = Mode::Effort;
    Target hover;
    Target pressed;
    bool dragging = false;
    I64 dragIndex = -1;
    F32 scroll = 0.0f;
    bool scrollReset = false;
    U64 rowLimit = 1;
    F32 anchorWidth = 0.0f;
    F32 thumb = -1.0f;
    std::string localEffort;
    std::string pendingModel;
    std::function<void()> notify;

    PickerMetrics metrics(const Context& ctx) const {
        const F32 unit = Px(ctx, config.fontSize / Typography::FontSize);
        return {.unit = unit, .font = Px(ctx, config.fontSize), .gap = kGap * unit};
    }

    Mode activeMode() const {
        return config.efforts.empty() ? Mode::Models : mode;
    }

    U64 stopCount() const {
        return std::min<U64>(config.efforts.size(), kMaxStops);
    }

    const std::string& modelLabel() const {
        for (const auto& option : config.models) {
            if (option.id == config.value) {
                return option.label;
            }
        }
        return config.placeholder;
    }

    const std::string& currentEffort() const {
        return localEffort.empty() ? config.effort : localEffort;
    }

    I64 effortIndex() const {
        if (dragging) {
            return dragIndex;
        }
        for (U64 i = 0; i < stopCount(); ++i) {
            if (config.efforts[i].id == currentEffort()) {
                return static_cast<I64>(i);
            }
        }
        return -1;
    }

    const std::string& effortLabel() const {
        static const std::string empty;
        const I64 index = effortIndex();
        return index >= 0 && static_cast<U64>(index) < stopCount() ? config.efforts[static_cast<U64>(index)].label
                                                                    : empty;
    }

    F32 measureText(const std::string& text, F32 size) {
        return text.empty() ? 0.0f : textMetrics.measure(Typography::BodyFont, text, size);
    }

    ColorRGBA<F32> color(const Context& ctx, const std::string& key) const {
        return WithAlpha(ctx.color(key), config.disabled ? kDisabledAlpha : 1.0f);
    }

    void changed() {
        if (notify) {
            notify();
        }
    }

    void setOpen(bool next) {
        if (open == next) {
            return;
        }
        open = next;
        mode = config.efforts.empty() ? Mode::Models : Mode::Effort;
        hover = {};
        pressed = {};
        dragging = false;
        dragIndex = -1;
        resetScroll();
        changed();
    }

    void resetScroll() {
        scroll = 0.0f;
        scrollReset = true;
    }

    void select(U64 row) {
        if (row >= config.models.size()) {
            return;
        }
        const std::string id = config.models[row].id;
        if (id == config.value) {
            if (config.efforts.empty()) {
                setOpen(false);
            } else {
                mode = Mode::Effort;
                hover = {};
                resetScroll();
                changed();
            }
            return;
        }
        pendingModel = id;
        if (config.onSelect) {
            config.onSelect(id);
        }
    }

    void commitEffort() {
        const I64 index = dragIndex;
        dragging = false;
        dragIndex = -1;
        changed();
        if (index < 0 || static_cast<U64>(index) >= stopCount()) {
            return;
        }
        const std::string id = config.efforts[static_cast<U64>(index)].id;
        if (id == currentEffort()) {
            return;
        }
        localEffort = id;
        if (config.onEffort) {
            config.onEffort(id);
        }
    }
};

struct PickerBody : public Component {
    PickerState& picker;
    Scroller scroller;
    Box background;
    Box highlight;
    Box rail;
    Box dots;
    Box knob;
    Label texts;
    Label icons;
    Rect modelRow;
    std::vector<F32> stops;
    Rect list;
    F32 rowHeight = 0.0f;
    F32 offset = 0.0f;
    std::optional<Extent2D<F32>> pointer;

    explicit PickerBody(PickerState& picker) : picker(picker) {
        setClipsChildren(true);
        addChild(scroller);
        addChild(background);
        addChild(highlight);
        addChild(rail);
        addChild(dots);
        addChild(knob);
        addChild(texts);
        addChild(icons);
    }

    void refresh() {
        invalidate(Dirty::Layout);
    }

    static F32 RowHeight(const PickerMetrics& m) {
        return m.font + kRowPadding * m.unit;
    }

    static U64 PoolRows(const Context& ctx, const PickerMetrics& m) {
        return std::max<U64>(1, static_cast<U64>(static_cast<F32>(ctx.framebufferSize.y) /
                                                 std::max(1.0f, RowHeight(m))));
    }

    U64 rowsShown(const Context& ctx, const PickerMetrics& m) const {
        return std::min({static_cast<U64>(picker.config.models.size()), PoolRows(ctx, m),
                         std::max<U64>(1, picker.rowLimit)});
    }

    F32 effortHeight(const PickerMetrics& m) const {
        return 2.0f * kEffortPadding * m.unit + m.font * (kTitleHeight + kModelRowHeight) + m.gap +
               kThumbSize * m.unit;
    }

    std::optional<U64> rowAt(const Extent2D<F32>& point) const {
        if (picker.activeMode() != Mode::Models || rowHeight <= 0.0f || !frame().contains(point.x, point.y) ||
            !list.contains(point.x, point.y)) {
            return std::nullopt;
        }
        const F32 y = point.y - list.y + offset;
        if (y < 0.0f) {
            return std::nullopt;
        }
        const U64 row = static_cast<U64>(y / rowHeight);
        if (row >= picker.config.models.size()) {
            return std::nullopt;
        }
        return row;
    }

    bool onTrack(const Extent2D<F32>& point) const {
        return picker.activeMode() == Mode::Effort && !stops.empty() && frame().contains(point.x, point.y) &&
               point.y >= modelRow.bottom();
    }

    Target targetAt(const Extent2D<F32>& point) const {
        if (picker.activeMode() == Mode::Effort) {
            if (modelRow.contains(point.x, point.y) && frame().contains(point.x, point.y)) {
                return {.kind = Target::Kind::ModelRow};
            }
            return {};
        }
        if (const auto row = rowAt(point)) {
            return {.kind = Target::Kind::Row, .row = *row};
        }
        return {};
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

    void setHover(Target next) {
        if (picker.hover != next) {
            picker.hover = next;
            refresh();
        }
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        picker.textMetrics.setWindow(ctx.render);
        const auto m = picker.metrics(ctx);
        if (picker.activeMode() == Mode::Effort) {
            const F32 width = std::max(picker.anchorWidth, kPanelWidth * m.unit);
            return FitExtent({std::ceil(width), std::ceil(effortHeight(m))}, available);
        }
        F32 widest = 0.0f;
        for (const auto& option : picker.config.models) {
            widest = std::max(widest, picker.measureText(option.label, m.font) + m.gap +
                                          picker.measureText(option.detail, m.font));
        }
        const F32 width = std::max({picker.anchorWidth, kPanelWidth * m.unit,
                                    widest + 2.0f * kListInset * m.unit + kMarkSpace * m.font});
        const F32 height = static_cast<F32>(rowsShown(ctx, m)) * RowHeight(m) + 2.0f * kListPadding * m.unit;
        return FitExtent({std::ceil(width), std::ceil(height)}, available);
    }

    void layoutEffort(const Context& ctx, const PickerMetrics& m, const Rect& bounds, U64 capacity) {
        auto& state = picker;
        const auto& config = state.config;
        const F32 pad = kEffortPadding * m.unit;
        const F32 thumbSize = kThumbSize * m.unit;
        const F32 trackHeight = kTrackHeight * m.unit;
        const Rect title = {bounds.x, bounds.y + pad, bounds.width, m.font * kTitleHeight};
        modelRow = {bounds.x + pad, title.bottom(), std::max(0.0f, bounds.width - 2.0f * pad),
                    m.font * kModelRowHeight};
        const F32 trackCenter = modelRow.bottom() + m.gap + thumbSize * 0.5f;
        const Rect track = {bounds.x + pad, trackCenter - trackHeight * 0.5f,
                            std::max(0.0f, bounds.width - 2.0f * pad), trackHeight};
        const F32 inset = trackHeight * 0.5f;
        const F32 span = std::max(0.0f, track.width - 2.0f * inset);
        const U64 count = state.stopCount();
        stops.clear();
        for (U64 i = 0; i < count; ++i) {
            stops.push_back(track.x + inset + StopFraction(i, count) * span);
        }
        list = {};
        rowHeight = 0.0f;
        offset = 0.0f;

        const I64 index = state.effortIndex();
        if (index < 0) {
            state.thumb = -1.0f;
        } else {
            const F32 target = StopFraction(static_cast<U64>(index), count);
            if (state.thumb < 0.0f) {
                state.thumb = target;
            } else {
                state.thumb += (target - state.thumb) * (1.0f - std::exp(-ctx.deltaTime * kThumbSpeed));
            }
            if (std::abs(target - state.thumb) * span < Px(ctx, kThumbSnap)) {
                state.thumb = target;
            } else {
                requestFrame();
            }
        }
        const bool hasThumb = state.thumb >= 0.0f;
        const F32 thumbX = track.x + inset + std::max(0.0f, state.thumb) * span;

        const bool hovered = state.hover.kind == Target::Kind::ModelRow;
        const auto rowColor = state.color(ctx, hovered ? config.textColorKey : config.detailColorKey);
        const std::string& effort = state.effortLabel();
        const std::string& model = state.modelLabel();
        const F32 modelWidth = state.measureText(model, m.font);
        const F32 chevron = kChevronScale * m.font;
        const F32 chevronGap = 0.5f * m.gap;
        const F32 modelX = std::round(modelRow.x + (modelRow.width - modelWidth - chevronGap - chevron) * 0.5f);

        highlight.update({
            .instances = {{.rect = modelRow, .visible = hovered && !modelRow.empty(),
                           .backgroundColor = state.color(ctx, config.rowHoveredColorKey)}},
            .clip = bounds,
            .cornerRadius = modelRow.height * 0.5f,
        });
        rail.update({
            .instances = {
                {.rect = track, .visible = !track.empty(), .backgroundColor = state.color(ctx, config.trackColorKey)},
                {.rect = {track.x, track.y, std::max(0.0f, thumbX + inset - track.x), track.height},
                 .visible = hasThumb && !track.empty(), .backgroundColor = state.color(ctx, config.accentColorKey)},
            },
            .clip = bounds,
            .cornerRadius = inset,
        });
        const F32 dot = kStopSize * m.unit;
        const auto dotColor = WithAlpha(state.color(ctx, config.detailColorKey), kStopAlpha);
        std::vector<Box::Instance> marks;
        for (const F32 x : stops) {
            marks.push_back({.rect = {x - dot * 0.5f, track.y + (track.height - dot) * 0.5f, dot, dot},
                             .visible = !hasThumb || x > thumbX + thumbSize * 0.5f,
                             .backgroundColor = dotColor});
        }
        dots.update({.instances = std::move(marks), .clip = bounds, .cornerRadius = dot * 0.5f, .capacity = kMaxStops});
        knob.update({
            .instances = {{.rect = {thumbX - thumbSize * 0.5f, trackCenter - thumbSize * 0.5f, thumbSize, thumbSize},
                           .visible = hasThumb, .backgroundColor = state.color(ctx, config.thumbColorKey)}},
            .clip = bounds,
            .cornerRadius = thumbSize * 0.5f,
        });
        texts.update({
            .instances = {
                {.rect = title, .str = effort.empty() ? config.effortPlaceholder : effort,
                 .visible = !title.empty(),
                 .color = state.color(ctx, effort.empty() ? config.detailColorKey : config.accentColorKey),
                 .fontSize = m.font * kTitleScale, .alignment = {1, 1}},
                {.rect = {modelX, modelRow.y, modelWidth, modelRow.height}, .str = model,
                 .visible = !modelRow.empty(), .color = rowColor, .fontSize = m.font, .alignment = {0, 1}},
            },
            .clip = bounds,
            .fontName = Typography::BodyFont,
            .maxCharacters = kMaxCharacters,
            .capacity = capacity,
        });
        icons.update({
            .instances = {{.rect = {modelX + modelWidth + chevronGap, modelRow.y, chevron, modelRow.height},
                           .str = ICON_LC_CHEVRON_RIGHT, .visible = !modelRow.empty(), .color = rowColor,
                           .fontSize = chevron, .alignment = {0, 1}}},
            .clip = bounds,
            .fontName = Typography::IconFont,
            .maxCharacters = 4,
            .capacity = 1,
        });
    }

    void layoutModels(const Context& ctx, const PickerMetrics& m, const Rect& bounds, U64 capacity) {
        auto& state = picker;
        const auto& config = state.config;
        const F32 pad = kListPadding * m.unit;
        const F32 inset = kListInset * m.unit;
        const U64 rows = config.models.size();
        rowHeight = RowHeight(m);
        const F32 content = static_cast<F32>(rows) * rowHeight;
        list = {bounds.x, bounds.y + pad, bounds.width, static_cast<F32>(rowsShown(ctx, m)) * rowHeight};
        state.scroll = std::clamp(state.scroll, 0.0f, std::max(0.0f, content - list.height));
        scroller.update({
            .contentHeight = content,
            .scrollY = state.scroll,
            .scrollbar = false,
            .wheel = false,
            .wheelStep = rowHeight,
            .onScrollY = [this](F32 next) {
                picker.scroll = next;
                refresh();
            },
        });
        layoutChild(ctx, scroller, list);
        offset = state.scroll + scroller.overscroll(ctx).y;
        if (pointer.has_value()) {
            state.hover = targetAt(*pointer);
        }
        modelRow = {};
        stops.clear();

        const auto textColor = state.color(ctx, config.textColorKey);
        const auto detailColor = state.color(ctx, config.detailColorKey);
        std::vector<Label::Instance> labels;
        Label::Instance check = {.visible = false};
        const U64 first = static_cast<U64>(std::max(0.0f, std::floor(offset / rowHeight)));
        const U64 last = std::min<U64>(rows, static_cast<U64>(std::max(0.0f, std::ceil((offset + list.height) / rowHeight))));
        for (U64 i = first; i < last; ++i) {
            const auto& option = config.models[i];
            const Rect row = {bounds.x + inset, list.y + static_cast<F32>(i) * rowHeight - offset,
                              std::max(0.0f, bounds.width - 2.0f * inset), rowHeight};
            const F32 labelWidth = state.measureText(option.label, m.font);
            labels.push_back({.rect = row, .str = option.label, .color = textColor, .fontSize = m.font,
                              .alignment = {0, 1}});
            labels.push_back({.rect = {row.x + labelWidth + m.gap, row.y,
                                       std::max(0.0f, row.width - labelWidth - m.gap - kCheckSpace * m.font),
                                       row.height},
                              .str = option.detail, .visible = !option.detail.empty(), .color = detailColor,
                              .fontSize = m.font, .alignment = {0, 1}});
            if (option.id == config.value) {
                check = {.rect = {row.right() - m.font, row.y, m.font, row.height}, .str = ICON_LC_CHECK,
                         .color = textColor, .fontSize = m.font * kCheckScale, .alignment = {1, 1}};
            }
        }
        const bool hovered = state.hover.kind == Target::Kind::Row && state.hover.row >= first && state.hover.row < last;
        const F32 hoverTop = list.y + static_cast<F32>(hovered ? state.hover.row : 0) * rowHeight - offset;
        highlight.update({
            .instances = {{.rect = {bounds.x + pad, hoverTop, std::max(0.0f, bounds.width - 2.0f * pad), rowHeight},
                           .visible = hovered, .backgroundColor = state.color(ctx, config.rowHoveredColorKey)}},
            .clip = list,
            .cornerRadius = std::max(0.0f, Px(ctx, config.popupRounding) - pad),
        });
        rail.update({.clip = bounds, .cornerRadius = kTrackHeight * m.unit * 0.5f});
        dots.update({.clip = bounds, .cornerRadius = kStopSize * m.unit * 0.5f, .capacity = kMaxStops});
        knob.update({.clip = bounds, .cornerRadius = kThumbSize * m.unit * 0.5f});
        texts.update({
            .instances = std::move(labels),
            .clip = list,
            .fontName = Typography::BodyFont,
            .maxCharacters = kMaxCharacters,
            .capacity = capacity,
        });
        icons.update({
            .instances = {std::move(check)},
            .clip = list,
            .fontName = Typography::IconFont,
            .maxCharacters = 4,
            .capacity = 1,
        });
    }

    void layout(const Context& ctx) override {
        picker.textMetrics.setWindow(ctx.render);
        const auto& config = picker.config;
        const auto m = picker.metrics(ctx);
        const Rect bounds = frame();
        background.update({
            .instances = {{.rect = bounds, .visible = !bounds.empty(),
                           .backgroundColor = picker.color(ctx, config.popupColorKey)}},
            .clip = bounds,
            .cornerRadius = Px(ctx, config.popupRounding),
            .borderWidth = Px(ctx, 1.0f),
            .borderColor = picker.color(ctx, config.popupBorderColorKey),
            .strokeFringe = true,
        });
        const U64 capacity = 2 * (PoolRows(ctx, m) + 1);
        if (picker.scrollReset) {
            picker.scrollReset = false;
            scroller.reset();
        }
        if (picker.activeMode() == Mode::Effort) {
            layoutEffort(ctx, m, bounds, capacity);
            scroller.update({.scrollbar = false, .wheel = false});
            layoutChild(ctx, scroller, {bounds.x, bounds.y, 0.0f, 0.0f});
        } else {
            layoutModels(ctx, m, bounds, capacity);
        }
        layoutChild(ctx, background, bounds);
        layoutChild(ctx, highlight, bounds);
        layoutChild(ctx, rail, bounds);
        layoutChild(ctx, dots, bounds);
        layoutChild(ctx, knob, bounds);
        layoutChild(ctx, texts, bounds);
        layoutChild(ctx, icons, bounds);
    }

    bool mouseEvent(const MouseEvent& event) override {
        auto& state = picker;
        if (state.config.disabled) {
            return false;
        }
        const auto& point = event.position;
        const bool inside = frame().contains(point.x, point.y);
        switch (event.type) {
            case MouseEventType::Move: {
                if (state.dragging) {
                    requestCursor(SurfaceCursor::Hand);
                    const I64 next = nearestStop(point.x);
                    if (next != state.dragIndex) {
                        state.dragIndex = next;
                        state.changed();
                    }
                    return true;
                }
                pointer = point;
                const Target next = targetAt(point);
                setHover(next);
                return next.kind != Target::Kind::None;
            }
            case MouseEventType::Leave:
                pointer.reset();
                state.pressed = {};
                setHover({});
                return false;
            case MouseEventType::Cancel:
                pointer.reset();
                state.pressed = {};
                if (state.dragging) {
                    state.dragging = false;
                    state.dragIndex = -1;
                    state.changed();
                }
                setHover({});
                return false;
            case MouseEventType::Scroll: {
                if (!inside || state.activeMode() != Mode::Models || rowHeight <= 0.0f) {
                    return inside;
                }
                pointer = point;
                scroller.wheel(event);
                refresh();
                return true;
            }
            case MouseEventType::Click: {
                if (event.button != MouseButton::Left) {
                    return false;
                }
                if (onTrack(point)) {
                    state.dragging = true;
                    state.dragIndex = nearestStop(point.x);
                    state.changed();
                    return true;
                }
                state.pressed = targetAt(point);
                return inside;
            }
            case MouseEventType::Release: {
                if (event.button != MouseButton::Left) {
                    return false;
                }
                if (state.dragging) {
                    state.dragIndex = nearestStop(point.x);
                    state.commitEffort();
                    return true;
                }
                const Target target = targetAt(point);
                const bool activate = target.kind != Target::Kind::None && target == state.pressed;
                state.pressed = {};
                if (!activate) {
                    return inside;
                }
                if (target.kind == Target::Kind::ModelRow) {
                    state.mode = Mode::Models;
                    state.resetScroll();
                    state.hover = {};
                    state.changed();
                } else {
                    state.select(target.row);
                }
                return true;
            }
            default:
                return false;
        }
    }

    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override {
        if (!frame().contains(point.x, point.y)) {
            return std::nullopt;
        }
        if (targetAt(point).kind != Target::Kind::None || onTrack(point)) {
            return SurfaceCursor::Hand;
        }
        return SurfaceCursor::Default;
    }
};

}  // namespace

struct ModelPicker::Impl {
    PickerState state;
    Box field;
    Label fieldLabels;
    Polyline caret;
    PickerBody body{state};
};

ModelPicker::ModelPicker() {
    impl = std::make_unique<Impl>();
    impl->state.notify = [this]() {
        invalidate(Dirty::Layout);
        if (impl->state.open) {
            impl->body.refresh();
        }
    };
    addChild(impl->field);
    addChild(impl->fieldLabels);
    addChild(impl->caret);
}

ModelPicker::~ModelPicker() = default;

void ModelPicker::update(Config config) {
    auto& state = impl->state;
    const bool changed = config != state.config;
    const bool selected = !state.pendingModel.empty() && config.value == state.pendingModel;
    const bool reset = config.disabled || config.value != state.config.value || config.efforts != state.config.efforts;
    const bool effortChanged = config.effort != state.config.effort || config.value != state.config.value;
    state.config = std::move(config);
    if (changed) {
        state.changed();
    }
    if (effortChanged) {
        state.localEffort.clear();
    }
    if (state.config.disabled) {
        state.hovered = false;
    }
    if (selected) {
        state.pendingModel.clear();
        state.thumb = -1.0f;
        if (state.config.efforts.empty()) {
            state.setOpen(false);
        } else {
            state.mode = Mode::Effort;
            state.hover = {};
            state.resetScroll();
            state.pressed = {};
        }
    } else if (reset) {
        state.pendingModel.clear();
        state.setOpen(false);
    }
}

Extent2D<F32> ModelPicker::measure(const Context& ctx, Extent2D<F32> available) {
    auto& state = impl->state;
    state.textMetrics.setWindow(ctx.render);
    const auto m = state.metrics(ctx);
    F32 width = state.measureText(state.modelLabel(), m.font);
    if (!state.effortLabel().empty()) {
        width += m.gap + state.measureText(state.effortLabel(), m.font);
    }
    width += 2.0f * kFieldPaddingX * m.unit + 0.5f * m.gap + kCaretWidth * m.unit;
    return FitExtent({std::ceil(width), m.font + 2.0f * kFieldPaddingY * m.unit}, available);
}

void ModelPicker::layout(const Context& ctx) {
    auto& state = impl->state;
    const auto& config = state.config;
    state.textMetrics.setWindow(ctx.render);
    const auto m = state.metrics(ctx);
    const Rect bounds = frame();
    const bool visible = !bounds.empty();
    const F32 padX = kFieldPaddingX * m.unit;
    const F32 caretWidth = kCaretWidth * m.unit;
    const std::string& model = state.modelLabel();
    const std::string& effort = state.effortLabel();
    const Rect textRect = {bounds.x + padX, bounds.y,
                           std::max(0.0f, bounds.width - 2.0f * padX - caretWidth - 0.5f * m.gap), bounds.height};
    const F32 effortX = std::min(textRect.right(), textRect.x + state.measureText(model, m.font) + m.gap);
    const std::string& fill = config.disabled ? config.colorKey
                              : state.open    ? config.activeColorKey
                              : state.hovered ? config.hoveredColorKey
                                              : config.colorKey;

    impl->field.update({
        .instances = {{.rect = bounds, .visible = visible, .backgroundColor = state.color(ctx, fill)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, config.rounding),
    });
    impl->fieldLabels.update({
        .instances = {
            {.rect = textRect, .str = model, .visible = visible, .color = state.color(ctx, config.textColorKey),
             .fontSize = m.font, .alignment = {0, 1}},
            {.rect = {effortX, bounds.y, std::max(0.0f, textRect.right() - effortX), bounds.height}, .str = effort,
             .visible = visible && !effort.empty(), .color = state.color(ctx, config.detailColorKey),
             .fontSize = m.font, .alignment = {0, 1}},
        },
        .clip = textRect,
        .fontName = Typography::BodyFont,
        .maxCharacters = kMaxCharacters,
        .capacity = 2,
    });
    const F32 halfLength = 0.5f * kCaretLength * m.unit;
    const F32 halfDepth = 0.5f * kCaretDepth * m.unit;
    const Extent2D<F32> center = {bounds.right() - padX - 0.5f * caretWidth,
                                  bounds.y + kCaretDrop * m.unit + 0.5f * bounds.height};
    impl->caret.update({
        .instances = {{
            .points = {
                {center.x - halfLength, center.y - halfDepth},
                {center.x + halfLength, center.y - halfDepth},
                {center.x, center.y + halfDepth},
            },
            .color = state.color(ctx, config.detailColorKey),
            .filled = true,
            .visible = visible,
        }},
        .clip = bounds,
        .fringe = Fringe(ctx),
    });
    layoutChild(ctx, impl->field, bounds);
    layoutChild(ctx, impl->fieldLabels, bounds);
    layoutChild(ctx, impl->caret, bounds);

    if (!state.open || config.disabled || !visible) {
        return;
    }
    const F32 above = bounds.y + ctx.translation.y;
    const F32 below = static_cast<F32>(ctx.framebufferSize.y) - (bounds.bottom() + ctx.translation.y);
    const F32 space = std::max(above, below) - Px(ctx, kItemSpacing) - 2.0f * kListPadding * m.unit;
    const U64 rowLimit = static_cast<U64>(std::max(0.0f, space) / PickerBody::RowHeight(m));
    if (rowLimit != state.rowLimit || bounds.width != state.anchorWidth) {
        state.rowLimit = rowLimit;
        state.anchorWidth = bounds.width;
        impl->body.refresh();
    }
    RequestPopup(ctx, {
        .content = &impl->body,
        .anchor = bounds,
        .placement = PopupPlacement::BelowEnd,
        .onDismiss = [state = &impl->state]() {
            state->setOpen(false);
        },
        .passthrough = bounds,
    });
}

void ModelPicker::wentOffscreen(U64) {
    impl->state.setOpen(false);
}

bool ModelPicker::mouseEvent(const MouseEvent& event) {
    auto& state = impl->state;
    if (state.config.disabled) {
        return false;
    }
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != state.hovered) {
                state.hovered = inside;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (state.hovered) {
                state.hovered = false;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (event.button != MouseButton::Left || !inside) {
                return false;
            }
            state.setOpen(!state.open);
            return true;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> ModelPicker::cursorAt(const Extent2D<F32>& point) const {
    if (impl->state.config.disabled || !frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

}  // namespace Jetstream::Sakura
