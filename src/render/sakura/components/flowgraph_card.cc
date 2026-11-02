#include <jetstream/render/sakura/components/flowgraph_card.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../detail/helpers.hh"
#include "../detail/text_metrics.hh"
#include "../metrics.hh"
#include "tools/text.hh"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kDesignWidth = 320.0f;
constexpr F32 kDesignHeight = 180.0f;
constexpr F32 kDesignPad = 8.0f;
constexpr F32 kNodeWidth = 64.0f;
constexpr F32 kNodeHeight = 28.0f;
constexpr F32 kNodeRounding = 8.0f;
constexpr F32 kColumnGap = 40.0f;
constexpr F32 kRowGap = 40.0f;
constexpr F32 kRowPitch = kNodeHeight + kRowGap;
constexpr F32 kLabelFont = 10.0f;
constexpr F32 kLabelPadding = 5.0f;
constexpr F32 kDotPitch = 16.0f;
constexpr F32 kDotSize = 1.6f;
constexpr F32 kDotAlpha = 0.56f;
constexpr F32 kNodeFill = 0.12f;
constexpr F32 kNodeFillHovered = 0.22f;
constexpr F32 kStrokeAlpha = 0.7f;
constexpr F32 kDividerAlpha = 0.6f;
constexpr F32 kHoverLift = 3.0f;
constexpr F64 kHoverDuration = 0.25;
constexpr F64 kMaxHoverStep = 0.1;
constexpr F32 kHoverBorderAlpha = 0.7f;
constexpr F32 kCategoryText = 12.5f;
constexpr F32 kTitleText = 24.0f;
constexpr F32 kDescriptionText = 16.0f;
constexpr F32 kSectionGap = 6.0f;
constexpr F32 kCategoryGap = 2.0f;
constexpr U64 kMaxNodes = 16;
constexpr U64 kMaxEdges = 2 * kMaxNodes;
constexpr U64 kWireSegments = 16;
constexpr U64 kWireVerticesPerPoint = 4;
constexpr U64 kWirePoints = kMaxEdges * 4 * (kWireSegments + 1);
constexpr U64 kWireCapacity = kWirePoints * kWireVerticesPerPoint;
constexpr U64 kDotColumns = static_cast<U64>(kDesignWidth / kDotPitch) + 1;
constexpr U64 kDotRows = static_cast<U64>(kDesignHeight / kDotPitch) + 1;
constexpr const char* kPreviewSurface = "background";

}  // namespace

struct FlowgraphCard::Impl {
    struct Diagram {
        struct Node {
            std::string id;
            std::string label;
        };

        struct Box {
            std::string label;
            F32 x = 0.0f;
            F32 y = 0.0f;
        };

        struct Slots {
            std::vector<std::vector<U64>> columns;
            std::vector<I64> slotColumn;
            std::vector<std::vector<U64>> inputs;
            std::vector<std::vector<U64>> outputs;
            std::vector<std::optional<F32>> top;
            std::vector<F32> left;
            std::vector<F32> right;
            F32 minTop = 0.0f;
            F32 maxTop = 0.0f;

            U64 add(I64 column) {
                const U64 slot = slotColumn.size();
                if (columns.size() <= static_cast<U64>(column)) {
                    columns.resize(static_cast<U64>(column) + 1);
                }
                columns[static_cast<U64>(column)].push_back(slot);
                slotColumn.push_back(column);
                inputs.emplace_back();
                outputs.emplace_back();
                return slot;
            }

            void link(const std::vector<U64>& path) {
                for (U64 step = 1; step < path.size(); ++step) {
                    outputs[path[step - 1]].push_back(path[step]);
                    inputs[path[step]].push_back(path[step - 1]);
                }
            }

            std::optional<F32> mean(const std::vector<U64>& ids, bool unmergedOnly) const {
                F32 sum = 0.0f;
                U64 used = 0;
                for (const U64 id : ids) {
                    if ((!unmergedOnly || inputs[id].size() == 1) && top[id].has_value()) {
                        sum += *top[id];
                        ++used;
                    }
                }
                return used > 0 ? std::optional<F32>(sum / static_cast<F32>(used)) : std::nullopt;
            }

            template <typename Desired>
            void placeColumn(const std::vector<U64>& column, const Desired& desired) {
                F32 floor = -std::numeric_limits<F32>::infinity();
                for (const U64 slot : column) {
                    const F32 fallback = top[slot].value_or(floor + kRowPitch);
                    const F32 y = std::max(desired(slot).value_or(fallback), floor + kRowPitch);
                    top[slot] = std::isfinite(y) ? y : 0.0f;
                    floor = *top[slot];
                }
            }

            void sweepForward() {
                const F32 infinity = std::numeric_limits<F32>::infinity();
                for (U64 index = 0; index < columns.size(); ++index) {
                    auto& column = columns[index];
                    if (index > 0) {
                        std::stable_sort(column.begin(), column.end(), [&](U64 a, U64 b) {
                            return mean(inputs[a], false).value_or(infinity) < mean(inputs[b], false).value_or(infinity);
                        });
                    }
                    std::vector<std::optional<F32>> wanted(slotColumn.size());
                    for (U64 start = 0; start < column.size();) {
                        const auto base = mean(inputs[column[start]], false);
                        U64 end = start + 1;
                        while (base.has_value() && end < column.size() && mean(inputs[column[end]], false) == base) {
                            ++end;
                        }
                        if (base.has_value()) {
                            for (U64 i = start; i < end; ++i) {
                                const F32 offset =
                                    static_cast<F32>(i - start) - static_cast<F32>(end - start - 1) * 0.5f;
                                wanted[column[i]] = *base + offset * kRowPitch;
                            }
                        }
                        start = end;
                    }
                    placeColumn(column, [&](U64 slot) {
                        return wanted[slot];
                    });
                }
            }

            void sweepBackward() {
                for (U64 index = columns.size(); index-- > 0;) {
                    placeColumn(columns[index], [&](U64 slot) {
                        return mean(outputs[slot], true);
                    });
                }
            }

            F32 spread(U64 nodeCount) {
                minTop = std::numeric_limits<F32>::infinity();
                maxTop = -std::numeric_limits<F32>::infinity();
                for (const auto& value : top) {
                    minTop = std::min(minTop, value.value_or(0.0f));
                    maxTop = std::max(maxTop, value.value_or(0.0f));
                }
                left.assign(columns.size(), 0.0f);
                right.assign(columns.size(), 0.0f);
                F32 x = 0.0f;
                for (U64 index = 0; index < columns.size(); ++index) {
                    const bool hasNode = std::any_of(columns[index].begin(), columns[index].end(), [&](U64 slot) {
                        return slot < nodeCount;
                    });
                    const F32 columnWidth = hasNode ? kNodeWidth : 0.0f;
                    left[index] = x;
                    right[index] = x + columnWidth;
                    x += columnWidth + kColumnGap;
                }
                return x - kColumnGap;
            }

            F32 centerY(U64 slot) const {
                return top[slot].value_or(0.0f) - minTop + kNodeHeight * 0.5f;
            }
        };

        std::vector<Node> nodes;
        std::vector<std::pair<U64, U64>> edges;
        std::vector<Box> boxes;
        std::vector<std::vector<Extent2D<F32>>> wires;
        F32 width = 0.0f;
        F32 height = 0.0f;

        std::optional<U64> node(std::string_view source) {
            const std::string token = Jetstream::Text::TrimBlank(std::string(source));
            const U64 open = token.find('[');
            const std::string id = Jetstream::Text::TrimBlank(token.substr(0, open));
            std::string label;
            if (open != std::string::npos) {
                const U64 close = token.find(']', open);
                const U64 length = close == std::string::npos ? close : close - open - 1;
                label = Jetstream::Text::TrimBlank(token.substr(open + 1, length));
            }
            if (id.empty()) {
                return std::nullopt;
            }
            for (U64 index = 0; index < nodes.size(); ++index) {
                if (nodes[index].id == id) {
                    if (nodes[index].label.empty()) {
                        nodes[index].label = label;
                    }
                    return index;
                }
            }
            if (nodes.size() >= kMaxNodes) {
                return std::nullopt;
            }
            nodes.push_back({.id = id, .label = label});
            return nodes.size() - 1;
        }

        void chain(std::string_view line) {
            std::optional<U64> previous;
            while (true) {
                const U64 arrow = line.find("-->");
                const auto current = node(line.substr(0, arrow));
                if (!current) {
                    return;
                }
                if (previous && *previous != *current && edges.size() < kMaxEdges &&
                    std::find(edges.begin(), edges.end(), std::pair{*previous, *current}) == edges.end()) {
                    edges.push_back({*previous, *current});
                }
                previous = current;
                if (arrow == std::string_view::npos) {
                    return;
                }
                line = line.substr(arrow + 3);
            }
        }

        static Diagram Parse(const std::string& source) {
            Diagram diagram;
            std::string_view rest = source;
            while (!rest.empty()) {
                const U64 newline = rest.find('\n');
                const std::string line = Jetstream::Text::TrimBlank(std::string(rest.substr(0, newline)));
                rest = newline == std::string_view::npos ? std::string_view{} : rest.substr(newline + 1);
                if (!line.empty() && !line.starts_with("%%")) {
                    diagram.chain(line);
                }
            }
            for (auto& node : diagram.nodes) {
                if (node.label.empty()) {
                    node.label = node.id;
                }
            }
            diagram.place();
            return diagram;
        }

        std::vector<I64> rankColumns() const {
            const U64 count = nodes.size();
            std::vector<std::vector<U64>> inputs(count);
            std::vector<std::vector<U64>> outputs(count);
            for (const auto& [from, to] : edges) {
                outputs[from].push_back(to);
                inputs[to].push_back(from);
            }
            std::vector<U64> pending(count);
            std::deque<U64> ready;
            for (U64 index = 0; index < count; ++index) {
                pending[index] = inputs[index].size();
                if (pending[index] == 0) {
                    ready.push_back(index);
                }
            }
            std::vector<I64> columns(count, -1);
            while (!ready.empty()) {
                const U64 id = ready.front();
                ready.pop_front();
                I64 column = 0;
                for (const U64 input : inputs[id]) {
                    column = std::max(column, columns[input] + 1);
                }
                columns[id] = column;
                for (const U64 output : outputs[id]) {
                    if (--pending[output] == 0) {
                        ready.push_back(output);
                    }
                }
            }
            for (auto& column : columns) {
                column = std::max<I64>(column, 0);
            }
            return columns;
        }

        std::vector<I64> assignColumns() const {
            auto columns = rankColumns();
            for (U64 index = 0; index < nodes.size(); ++index) {
                bool fed = false;
                I64 lowest = std::numeric_limits<I64>::max();
                for (const auto& [from, to] : edges) {
                    fed |= to == index;
                    if (from == index) {
                        lowest = std::min(lowest, columns[to]);
                    }
                }
                if (!fed && lowest != std::numeric_limits<I64>::max()) {
                    columns[index] = std::max<I64>(0, lowest - 1);
                }
            }
            return columns;
        }

        std::vector<std::vector<U64>> addLanes(Slots& slots, const std::vector<I64>& columnOf) const {
            std::vector<std::vector<U64>> lanes(edges.size());
            for (U64 index = 0; index < edges.size(); ++index) {
                const auto [from, to] = edges[index];
                std::vector<U64> path = {from};
                for (I64 column = columnOf[from] + 1; column < columnOf[to]; ++column) {
                    const U64 slot = slots.add(column);
                    lanes[index].push_back(slot);
                    path.push_back(slot);
                }
                path.push_back(to);
                slots.link(path);
            }
            return lanes;
        }

        void route(const Slots& slots, const std::vector<I64>& columnOf, const std::vector<std::vector<U64>>& lanes) {
            U64 curves = 0;
            U64 straights = 0;
            for (const auto& lane : lanes) {
                curves += lane.size() + 1;
                straights += lane.size();
            }
            const U64 fixed = edges.size() + straights;
            const U64 spare = kWirePoints > fixed ? kWirePoints - fixed : 0;
            const U64 segments = std::clamp<U64>(curves > 0 ? spare / curves : kWireSegments, 1, kWireSegments);

            wires.clear();
            for (U64 index = 0; index < edges.size(); ++index) {
                const U64 from = edges[index].first;
                const U64 to = edges[index].second;
                std::vector<Extent2D<F32>> points = {
                    {slots.right[static_cast<U64>(columnOf[from])], slots.centerY(from)}};
                const auto curveTo = [&](Extent2D<F32> end) {
                    const Extent2D<F32> start = points.back();
                    const F32 bend = (end.x - start.x) * 0.5f;
                    auto curve = Polyline::Bezier(start, {start.x + bend, start.y}, {end.x - bend, end.y}, end,
                                                  segments);
                    points.insert(points.end(), curve.begin() + 1, curve.end());
                };
                for (const U64 lane : lanes[index]) {
                    const U64 column = static_cast<U64>(slots.slotColumn[lane]);
                    curveTo({slots.left[column], slots.centerY(lane)});
                    if (slots.right[column] != slots.left[column]) {
                        points.push_back({slots.right[column], slots.centerY(lane)});
                    }
                }
                curveTo({slots.left[static_cast<U64>(columnOf[to])], slots.centerY(to)});
                wires.push_back(std::move(points));
            }
        }

        void place() {
            const U64 count = nodes.size();
            if (count == 0) {
                return;
            }
            const auto columnOf = assignColumns();
            Slots slots;
            for (U64 index = 0; index < count; ++index) {
                slots.add(columnOf[index]);
            }
            const auto lanes = addLanes(slots, columnOf);
            slots.top.assign(slots.slotColumn.size(), std::nullopt);
            slots.sweepForward();
            slots.sweepBackward();

            width = slots.spread(count);
            height = slots.maxTop - slots.minTop + kNodeHeight;
            boxes.clear();
            for (U64 index = 0; index < count; ++index) {
                boxes.push_back({
                    .label = nodes[index].label,
                    .x = slots.left[static_cast<U64>(columnOf[index])],
                    .y = slots.top[index].value_or(0.0f) - slots.minTop,
                });
            }
            route(slots, columnOf, lanes);
        }
    };

    Config config;
    Diagram diagram;
    Box card;
    Box preview;
    Box divider;
    Box dots;
    Polyline wires;
    Box boxes;
    Label labels;
    Text category;
    Text title;
    Text accentTitle;
    Text description;
    TextMetrics textMetrics;
    bool hovered = false;
    bool pressed = false;
    F32 hover = 0.0f;
    F64 steppedAt = -1.0;

    void pushTitle() {
        const Text::Config base = {
            .str = config.title,
            .font = Text::Font::DisplaySemiBold,
            .wrapped = true,
            .scale = kTitleText / Typography::FontSize,
            .weight = kLabelWeight,
        };
        title.update(base);
        auto accent = base;
        accent.colorKey = config.accentColorKey;
        accent.opacity = Smoothstep(hover);
        accentTitle.update(accent);
    }

    bool advance(F64 time) {
        const F32 dt = steppedAt < 0.0 ? 0.0f : static_cast<F32>(std::clamp(time - steppedAt, 0.0, kMaxHoverStep));
        steppedAt = time;
        const F32 target = hovered ? 1.0f : 0.0f;
        const F32 step = dt / static_cast<F32>(kHoverDuration);
        hover = target > hover ? std::min(target, hover + step) : std::max(target, hover - step);
        if (hover == target) {
            steppedAt = -1.0;
            return false;
        }
        return true;
    }

    struct Metrics {
        F32 category = 0.0f;
        F32 title = 0.0f;
        F32 description = 0.0f;
        F32 height = 0.0f;
    };

    F32 previewHeight(F32 width) const {
        return config.diagram.empty() ? 0.0f : std::round(width * kDesignHeight / kDesignWidth);
    }

    Metrics metrics(FlowgraphCard& self, const Context& ctx, F32 width) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        const F32 inner = std::max(0.0f, width - 2.0f * Px(ctx, kCardPadX));
        Metrics result;
        result.category = self.measureChild(category, ctx, {inner, infinite}).y;
        result.title = self.measureChild(title, ctx, {inner, infinite}).y;
        result.description = config.description.empty() ? 0.0f
                                                       : self.measureChild(description, ctx, {inner, infinite}).y;
        const F32 gap = Px(ctx, kSectionGap);
        result.height = previewHeight(width) + Px(ctx, kCardPadX) + result.category + Px(ctx, kCategoryGap) +
                        result.title + gap + result.description + Px(ctx, kCardPadX);
        return result;
    }
};

FlowgraphCard::FlowgraphCard() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->card, &impl->preview, &impl->dots, &impl->divider, &impl->wires, &impl->boxes,
                   &impl->labels, &impl->category, &impl->title, &impl->accentTitle, &impl->description});
}

FlowgraphCard::~FlowgraphCard() = default;

void FlowgraphCard::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
        if (config.diagram != impl->config.diagram) {
            impl->diagram = Impl::Diagram::Parse(config.diagram);
        }
        if (config.id != impl->config.id) {
            impl->pressed = false;
        }
    }
    impl->config = std::move(config);
    impl->category.update({
        .str = Jetstream::Text::ToUpper(impl->config.category),
        .font = Text::Font::SansBold,
        .tone = Text::Tone::Secondary,
        .scale = kCategoryText / Typography::FontSize,
    });
    impl->pushTitle();
    impl->description.update({
        .str = impl->config.description,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
        .scale = kDescriptionText / Typography::FontSize,
        .lineHeight = 1.1f,
    });
}

Extent2D<F32> FlowgraphCard::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 width = FillLength(available.x);
    return {width, FitLength(impl->metrics(*this, CardSurface(ctx), width).height, available.y)};
}

void FlowgraphCard::layout(const Context& ctx) {
    auto& s = *impl;
    if (s.advance(ctx.time)) {
        requestFrame();
    }
    const F32 hover = Smoothstep(s.hover);
    s.pushTitle();
    const Rect slot = frame();
    const Rect bounds = {slot.x, slot.y - std::round(Px(ctx, kHoverLift) * hover), slot.width, slot.height};
    const Context surface = CardSurface(ctx);
    const auto metrics = s.metrics(*this, surface, bounds.width);
    const F32 pad = Px(ctx, kCardPadX);
    const F32 inner = std::max(0.0f, bounds.width - 2.0f * pad);
    const auto accent = ctx.color(s.config.accentColorKey);
    const Rect preview = {bounds.x, bounds.y, bounds.width, std::min(bounds.height, s.previewHeight(bounds.width))};
    const auto backdrop = ctx.color(kPreviewSurface);

    s.card.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color(kCardSurface)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, kPanelRounding),
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = MixColor(ctx.color("setting_card_border"), WithAlpha(accent, kHoverBorderAlpha), hover),
        .strokeFringe = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.card, bounds);
    s.preview.update({
        .instances = {{.rect = Inset(preview, Hairline(ctx)), .visible = !preview.empty(), .backgroundColor = backdrop,
                       .corners = static_cast<U8>(Box::TopLeft | Box::TopRight)}},
        .clip = bounds,
        .cornerRadius = std::max(0.0f, Px(ctx, kPanelRounding) - Hairline(ctx)),
        .capacity = 1,
    });
    layoutChild(ctx, s.preview, bounds);

    const F32 outline = Px(ctx, 1.0f);
    const Rect inside = {preview.x + outline, preview.y + outline, std::max(0.0f, preview.width - 2.0f * outline),
                         std::max(0.0f, preview.height - outline)};
    const F32 rounding = std::max(0.0f, Px(ctx, kPanelRounding) - outline);
    const auto within = [&](Extent2D<F32> point) {
        if (!inside.contains(point.x, point.y)) {
            return false;
        }
        const F32 top = inside.y + rounding;
        if (point.y >= top) {
            return true;
        }
        const F32 cornerX = point.x < inside.x + rounding ? inside.x + rounding
                            : point.x > inside.right() - rounding ? inside.right() - rounding
                                                                   : point.x;
        const F32 dx = point.x - cornerX;
        const F32 dy = point.y - top;
        return dx * dx + dy * dy <= rounding * rounding;
    };

    const F32 unit = preview.width / kDesignWidth;
    std::vector<Box::Instance> dots;
    const F32 dot = kDotSize * unit;
    for (U64 row = 0; row < kDotRows; ++row) {
        for (U64 column = 0; column < kDotColumns; ++column) {
            const F32 x = preview.x + (static_cast<F32>(column) * kDotPitch + 1.0f) * unit - dot * 0.5f;
            const F32 y = preview.y + (static_cast<F32>(row) * kDotPitch + 1.0f) * unit - dot * 0.5f;
            const bool shown = !preview.empty() && within({x, y}) && within({x + dot, y}) &&
                               within({x, y + dot}) && within({x + dot, y + dot});
            dots.push_back({.rect = {x, y, dot, dot}, .visible = shown,
                            .backgroundColor = WithAlpha(ctx.color("border"), kDotAlpha)});
        }
    }
    s.dots.update({.instances = std::move(dots), .clip = inside, .cornerRadius = dot * 0.5f,
                   .capacity = kDotColumns * kDotRows});
    layoutChild(ctx, s.dots, bounds);

    const F32 line = Hairline(ctx);
    s.divider.update({
        .instances = {{.rect = {preview.x, preview.bottom() - line, preview.width, line}, .visible = !preview.empty(),
                       .backgroundColor = WithAlpha(ctx.color("border"), kDividerAlpha)}},
        .capacity = 1,
    });
    layoutChild(ctx, s.divider, bounds);

    const auto& diagram = s.diagram;
    const F32 fit = diagram.width > 0.0f && diagram.height > 0.0f
                        ? std::min({1.0f, (kDesignWidth - 2.0f * kDesignPad) / diagram.width,
                                    (kDesignHeight - 2.0f * kDesignPad) / diagram.height})
                        : 1.0f;
    const F32 scale = unit * fit;
    const F32 originX = preview.x + (kDesignWidth - diagram.width * fit) * 0.5f * unit;
    const F32 originY = preview.y + (kDesignHeight - diagram.height * fit) * 0.5f * unit;
    const auto toScreen = [&](Extent2D<F32> point) {
        return Extent2D<F32>{originX + point.x * scale, originY + point.y * scale};
    };
    const auto stroke = WithAlpha(accent, kStrokeAlpha);

    std::vector<Polyline::Instance> wires;
    for (const auto& path : diagram.wires) {
        std::vector<Extent2D<F32>> points;
        points.reserve(path.size());
        for (const auto& point : path) {
            points.push_back(toScreen(point));
        }
        wires.push_back({.points = std::move(points), .color = stroke, .thickness = Px(ctx, 1.0f)});
    }
    s.wires.update({.instances = std::move(wires), .clip = inside, .fringe = Fringe(ctx), .capacity = kWireCapacity});
    layoutChild(ctx, s.wires, bounds);

    std::vector<Box::Instance> boxes;
    std::vector<Label::Instance> labels;
    const auto fill = BlendOver(backdrop, WithAlpha(accent, kNodeFill + (kNodeFillHovered - kNodeFill) * hover));
    s.textMetrics.setWindow(ctx.render);
    const F32 labelFont = kLabelFont * scale;
    const F32 labelWidth = std::max(0.0f, (kNodeWidth - 2.0f * kLabelPadding) * scale);
    for (const auto& box : diagram.boxes) {
        const auto corner = toScreen({box.x, box.y});
        const Rect rect = {corner.x, corner.y, kNodeWidth * scale, kNodeHeight * scale};
        boxes.push_back({.rect = rect, .backgroundColor = fill});
        labels.push_back({
            .rect = rect,
            .str = s.textMetrics.ellipsize(Typography::MonoFont, box.label, labelFont, labelWidth),
            .visible = !box.label.empty() && !rect.empty(),
            .color = accent,
            .fontSize = labelFont,
            .alignment = {1, 1},
        });
    }
    s.boxes.update({
        .instances = std::move(boxes),
        .clip = inside,
        .cornerRadius = kNodeRounding * scale,
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = stroke,
        .strokeFringe = true,
        .capacity = kMaxNodes,
    });
    layoutChild(ctx, s.boxes, bounds);
    s.labels.update({
        .instances = std::move(labels),
        .clip = inside,
        .fontName = Typography::MonoFont,
        .maxCharacters = 24,
        .capacity = kMaxNodes,
    });
    layoutChild(ctx, s.labels, bounds);

    const F32 gap = Px(ctx, kSectionGap);
    F32 y = preview.bottom() + pad;
    layoutChild(surface, s.category, {bounds.x + pad, y, inner, metrics.category}, bounds);
    y += metrics.category + Px(ctx, kCategoryGap);
    layoutChild(surface, s.title, {bounds.x + pad, y, inner, metrics.title}, bounds);
    layoutChild(surface, s.accentTitle, {bounds.x + pad, y, inner, metrics.title}, bounds);
    y += metrics.title + gap;
    layoutChild(surface, s.description, {bounds.x + pad, y, inner, metrics.description}, bounds);
}

bool FlowgraphCard::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != s.hovered) {
                s.hovered = inside;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (!inside || event.button != MouseButton::Left) {
                return false;
            }
            s.pressed = true;
            return true;
        case MouseEventType::Release:
            if (event.button != MouseButton::Left || !s.pressed) {
                return false;
            }
            s.pressed = false;
            if (inside && s.config.onOpen) {
                s.config.onOpen();
            }
            return true;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered || s.pressed) {
                s.hovered = false;
                s.pressed = false;
                invalidate(Dirty::Layout);
            }
            return false;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> FlowgraphCard::cursorAt(const Extent2D<F32>& point) const {
    if (frame().contains(point.x, point.y)) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
