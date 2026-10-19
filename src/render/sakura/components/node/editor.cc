#include <jetstream/render/sakura/components/node/editor.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/curves.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/tools/imgui_icons_ext.hh>

#include "common.hh"
#include "../../metrics.hh"
#include "../../detail/access.hh"

#include <algorithm>
#include <bit>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Jetstream::Sakura {

using namespace NodeStyle;

namespace {

constexpr F32 kMarqueeFill = 0.06f;
constexpr F32 kMarqueeBorder = 0.45f;
constexpr F32 kPadding = 8.0f;
constexpr F32 kPinRadius = 4.0f;
constexpr U64 kPinSegments = 16;
constexpr F32 kGridSpacing = 23.0f;
constexpr F32 kBorderThickness = 2.0f;
constexpr F32 kSelectedLighten = 0.25f;
constexpr F32 kCornerRounding = 12.0f;
constexpr F32 kEditorRounding = 10.0f;
constexpr F32 kLinkThickness = 1.5f;
constexpr F32 kLinkSegmentsPerLength = 0.2f;
constexpr U64 kLinkCapacity = 4096;
constexpr U64 kLinkMaxSegments = 256;
constexpr U64 kGridCapacity = 16384;
constexpr U64 kGripCapacity = 64;
constexpr U64 kChaseCapacity = 2560;
constexpr U64 kTextCapacity = 16;
constexpr std::array<F32, 10> kZoomSteps = {0.5f, 0.67f, 0.8f, 0.9f, 1.0f, 1.1f, 1.25f, 1.5f, 1.75f, 2.0f};
constexpr F32 kLinkHoverDistance = 10.0f;
constexpr F32 kPinHoverRadius = 10.0f;
constexpr F32 kTitleFont = BaseFont * 1.15f;
constexpr U64 kTimingCapacity = 8;
constexpr F32 kSubtitleFont = BaseFont * 0.75f;
constexpr F32 kSubtitlePull = 14.0f;
constexpr F32 kTitleGap = 8.0f;
constexpr F32 kTimingOffset = 4.0f;
constexpr F32 kLoadingMinHeight = 96.0f;
constexpr F32 kAutoPanSpeed = 1000.0f;
constexpr F32 kChevronDragTolerance = 3.0f;
constexpr F32 Pi = std::numbers::pi_v<F32>;

using NodeState = NodeEditor::NodeState;
using ResizeAxes = NodeEditor::ResizeAxes;

bool HasAxis(ResizeAxes axes, ResizeAxes axis) {
    return (static_cast<U8>(axes) & static_cast<U8>(axis)) != 0;
}

const char* StateColorKey(NodeState state) {
    switch (state) {
        case NodeState::Error:
            return "node_outline_error";
        case NodeState::Pending:
        case NodeState::Loading:
            return "node_outline_pending";
        default:
            return "node_outline";
    }
}

ColorRGBA<F32> Lighten(ColorRGBA<F32> color, F32 amount, F32 limit) {
    color.r = std::min(color.r + amount, limit);
    color.g = std::min(color.g + amount, limit);
    color.b = std::min(color.b + amount, limit);
    return color;
}

ColorRGBA<F32> Doubled(ColorRGBA<F32> color) {
    color.r = std::min(color.r * 2.0f, 1.0f);
    color.g = std::min(color.g * 2.0f, 1.0f);
    color.b = std::min(color.b * 2.0f, 1.0f);
    return color;
}

F32 Length(Extent2D<F32> v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

Extent2D<F32> ClosestPointOnSegment(Extent2D<F32> a, Extent2D<F32> b, Extent2D<F32> p) {
    const Extent2D<F32> ab = {b.x - a.x, b.y - a.y};
    const F32 lengthSquared = ab.x * ab.x + ab.y * ab.y;
    if (lengthSquared <= 0.0f) {
        return a;
    }
    const F32 t = std::clamp(((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / lengthSquared, 0.0f, 1.0f);
    return {a.x + ab.x * t, a.y + ab.y * t};
}

struct Bezier {
    Extent2D<F32> p0;
    Extent2D<F32> p1;
    Extent2D<F32> p2;
    Extent2D<F32> p3;
    U64 segments = 1;
};

Bezier LinkBezier(Extent2D<F32> start, Extent2D<F32> end, bool startIsInput, F32 segmentsPerLength) {
    if (startIsInput) {
        std::swap(start, end);
    }
    const F32 length = Length({end.x - start.x, end.y - start.y});
    const Extent2D<F32> offset = {0.25f * length, 0.0f};
    return {
        .p0 = start,
        .p1 = {start.x + offset.x, start.y},
        .p2 = {end.x - offset.x, end.y},
        .p3 = end,
        .segments = static_cast<U64>(std::max(static_cast<I64>(length * segmentsPerLength), static_cast<I64>(1))),
    };
}

F32 DistanceToBezier(const Bezier& curve, Extent2D<F32> point) {
    const auto points = Polyline::Bezier(curve.p0, curve.p1, curve.p2, curve.p3, curve.segments);
    F32 best = std::numeric_limits<F32>::max();
    for (U64 i = 1; i < points.size(); ++i) {
        const auto closest = ClosestPointOnSegment(points[i - 1], points[i], point);
        best = std::min(best, Length({closest.x - point.x, closest.y - point.y}));
    }
    return best;
}

Rect BezierBounds(const Bezier& curve, F32 expand) {
    F32 minX = std::min({curve.p0.x, curve.p1.x, curve.p2.x, curve.p3.x});
    F32 minY = std::min({curve.p0.y, curve.p1.y, curve.p2.y, curve.p3.y});
    F32 maxX = std::max({curve.p0.x, curve.p1.x, curve.p2.x, curve.p3.x});
    F32 maxY = std::max({curve.p0.y, curve.p1.y, curve.p2.y, curve.p3.y});
    return {minX - expand, minY - expand, (maxX - minX) + 2.0f * expand, (maxY - minY) + 2.0f * expand};
}

Extent2D<F32> GridToPx(const Rect& viewport, Extent2D<F32> offset, F32 ratio, Extent2D<F32> grid) {
    return {viewport.x + grid.x * ratio + offset.x, viewport.y + grid.y * ratio + offset.y};
}

Extent2D<F32> PointOnRoundedRect(const Rect& rect, F32 radius, F32 distance) {
    const F32 width = rect.width;
    const F32 height = rect.height;
    const F32 r = std::clamp(radius, 0.0f, std::min(width, height) * 0.5f);
    const F32 straightWidth = std::max(0.0f, width - 2.0f * r);
    const F32 straightHeight = std::max(0.0f, height - 2.0f * r);
    const F32 arcLength = Pi * r * 0.5f;
    const F32 perimeter = 2.0f * (straightWidth + straightHeight) + 2.0f * Pi * r;
    if (perimeter <= 0.0f) {
        return {rect.x, rect.y};
    }
    F32 d = std::fmod(distance, perimeter);
    if (d < 0.0f) {
        d += perimeter;
    }
    const auto arc = [r](Extent2D<F32> center, F32 start, F32 along) {
        const F32 angle = start + (r > 0.0f ? along / r : 0.0f);
        return Extent2D<F32>{center.x + std::cos(angle) * r, center.y + std::sin(angle) * r};
    };
    const F32 minX = rect.x;
    const F32 minY = rect.y;
    const F32 maxX = rect.right();
    const F32 maxY = rect.bottom();
    if (d <= straightWidth) {
        return {minX + r + d, minY};
    }
    d -= straightWidth;
    if (d <= arcLength) {
        return arc({maxX - r, minY + r}, -Pi * 0.5f, d);
    }
    d -= arcLength;
    if (d <= straightHeight) {
        return {maxX, minY + r + d};
    }
    d -= straightHeight;
    if (d <= arcLength) {
        return arc({maxX - r, maxY - r}, 0.0f, d);
    }
    d -= arcLength;
    if (d <= straightWidth) {
        return {maxX - r - d, maxY};
    }
    d -= straightWidth;
    if (d <= arcLength) {
        return arc({minX + r, maxY - r}, Pi * 0.5f, d);
    }
    d -= arcLength;
    if (d <= straightHeight) {
        return {minX, maxY - r - d};
    }
    d -= straightHeight;
    return arc({minX + r, minY + r}, Pi, d);
}

F32 RoundedRectPerimeter(const Rect& rect, F32 radius) {
    const F32 r = std::clamp(radius, 0.0f, std::min(rect.width, rect.height) * 0.5f);
    return 2.0f * (std::max(0.0f, rect.width - 2.0f * r) + std::max(0.0f, rect.height - 2.0f * r)) +
           2.0f * Pi * r;
}

}  // namespace

struct NodeFrame : public Component {
    struct PinLayout {
        Rect label;
        Extent2D<F32> center;
    };

    NodeEditor::Node spec;
    NodeContentStack* content = nullptr;

    Extent2D<F32> origin = {0.0f, 0.0f};
    Extent2D<F32> contentSize = {0.0f, 0.0f};
    Extent2D<F32> requested = {0.0f, 0.0f};
    bool hasRequested = false;
    std::optional<Extent2D<F32>> appliedGridPosition;

    Rect rect;
    F32 contentWidth = 0.0f;
    F32 contentTop = 0.0f;
    F32 contentHeight = 0.0f;
    F32 fixedHeight = 0.0f;
    F32 minimumHeight = 0.0f;
    bool hasContent = false;
    std::string displayTitle;
    F32 titleX = 0.0f;
    F32 subtitleX = 0.0f;
    Rect chevronRect;
    Rect skullRect;
    std::vector<PinLayout> pins;
    std::optional<NodeEditor::Geometry> lastGeometry;

    Box chrome;
    Polyline shapes;
    Polyline grip;
    Label header;
    Label texts;
    Label icons;
    Box loadingBorder;
    Polyline chase;
    Box badge;
    Label badgeText;
    Label timing;
    TextMetrics metrics;
    bool loadingMounted = false;
    struct Measured {
        U64 revision = 0;
        U64 epoch = 0;
        U64 fonts = 0;
        F32 ratio = 0.0f;
        Extent2D<F32> origin = {0.0f, 0.0f};
        Extent2D<F32> contentSize = {0.0f, 0.0f};

        bool operator==(const Measured&) const = default;
    };
    std::optional<Measured> measured;

    NodeFrame() {
        rebuildChildren();
    }

    void rebuildChildren() {
        clearChildren();
        addChild(chrome);
        addChild(shapes);
        if (content) {
            addChild(*content);
        }
        addChild(grip);
        addChild(header);
        addChild(texts);
        addChild(icons);
        addChild(timing);
        if (loadingMounted) {
            mountLoading();
        }
    }

    void mountLoading() {
        addChild(loadingBorder);
        addChild(chase);
        addChild(badge);
        addChild(badgeText);
    }

    void unmountLoading() {
        removeChild(loadingBorder);
        removeChild(chase);
        removeChild(badge);
        removeChild(badgeText);
    }

    void apply(NodeEditor::Node next) {
        const Extent2D<F32> previousOrigin = origin;
        const Extent2D<F32> previousSize = contentSize;
        const bool changed = next != spec;
        if (next.content != content || (content && !hasChild(*content))) {
            content = next.content;
            rebuildChildren();
        }
        const bool loading = next.state == NodeState::Loading;
        if (loading != loadingMounted) {
            loadingMounted = loading;
            if (loading) {
                mountLoading();
            } else {
                unmountLoading();
            }
        }

        const bool resizeX = HasAxis(next.resize, ResizeAxes::X);
        const bool resizeY = HasAxis(next.resize, ResizeAxes::Y);
        Extent2D<F32> wanted = next.dimensions;
        if (next.state == NodeState::Loading) {
            wanted.y = std::max(wanted.y, kLoadingMinHeight);
        }
        if (!hasRequested || wanted.x != requested.x || wanted.y != requested.y) {
            contentSize = wanted;
        }
        if (wanted.x <= 0.0f && !resizeX) {
            contentSize.x = 0.0f;
        }
        if (wanted.y <= 0.0f && !resizeY) {
            contentSize.y = 0.0f;
        }
        requested = wanted;
        hasRequested = true;

        if (next.gridPosition.has_value()) {
            if (!appliedGridPosition.has_value() ||
                appliedGridPosition->x != next.gridPosition->x ||
                appliedGridPosition->y != next.gridPosition->y) {
                origin = *next.gridPosition;
                appliedGridPosition = next.gridPosition;
            }
        } else {
            appliedGridPosition.reset();
        }

        spec = std::move(next);
        if (changed || origin != previousOrigin || contentSize != previousSize) {
            invalidate(Dirty::Layout);
        }
    }

    F32 effectiveMinimumHeight() const {
        return std::max(spec.minimumDimensions.y, minimumHeight);
    }

    void measureLayout(const Context& ctx) {
        const Measured key = {
            .revision = layoutRevision(),
            .epoch = Access::layoutEpoch(),
            .fonts = ctx.render ? ctx.render->fontRevision() : 0,
            .ratio = ctx.pixelRatio,
            .origin = origin,
            .contentSize = contentSize,
        };
        if (measured == key) {
            return;
        }
        invalidate(Dirty::Layout);
        metrics.setWindow(ctx.render);
        const F32 r = std::max(1e-3f, ctx.pixelRatio);

        const F32 width = std::max(spec.minimumDimensions.x, contentSize.x > 0.0f ? contentSize.x : spec.dimensions.x);
        contentWidth = width;
        F32 widest = width;

        F32 cursor = 0.0f;
        F32 lastBottom = 0.0f;
        bool any = false;

        displayTitle.clear();
        chevronRect = {};
        skullRect = {};
        if (spec.header) {
            const bool hasDiagnostic = spec.state != NodeState::Normal && !spec.diagnostic.empty();
            const bool hasChevron = spec.collapsible && spec.onToggleCollapse;
            const F32 iconWidth = NodeFont;
            F32 reserved = 0.0f;
            if (hasChevron) {
                reserved += iconWidth;
            }
            if (hasChevron && hasDiagnostic) {
                reserved += kTitleGap;
            }
            if (hasDiagnostic) {
                reserved += iconWidth;
            }
            if (reserved > 0.0f) {
                reserved += kTitleGap;
            }
            const F32 titleAvail = std::max(0.0f, width - reserved);
            const F32 centerSafe = std::max(0.0f, width - 2.0f * reserved);
            const F32 titleWidth = metrics.measure(HeaderFont, spec.title, kTitleFont * r) / r;
            displayTitle = spec.title;
            F32 zone = width;
            if (titleWidth > centerSafe) {
                if (titleWidth > titleAvail) {
                    displayTitle = Ellipsize(metrics, HeaderFont, spec.title, kTitleFont * r, titleAvail * r);
                }
                zone = titleAvail;
            }
            const F32 displayWidth = metrics.measure(HeaderFont, displayTitle, kTitleFont * r) / r;
            titleX = std::max(0.0f, (zone - displayWidth) * 0.5f);

            const F32 iconY = (kTitleFont - iconWidth) * 0.5f;
            if (hasChevron) {
                chevronRect = {width - iconWidth, iconY, iconWidth, iconWidth};
            }
            if (hasDiagnostic) {
                const F32 skullX = hasChevron ? chevronRect.x - kTitleGap - iconWidth : width - iconWidth;
                skullRect = {skullX, iconY, iconWidth, iconWidth};
            }

            const F32 subtitleWidth = metrics.measure(HeaderFont, spec.subtitle, kSubtitleFont * r) / r;
            subtitleX = std::max(0.0f, (width - subtitleWidth) * 0.5f);
            widest = std::max(widest, subtitleWidth);

            const F32 contentOrigin = kTitleFont + 2.0f * kPadding;
            const F32 subtitleTop = contentOrigin - kSubtitlePull;
            lastBottom = subtitleTop + kSubtitleFont;
            cursor = lastBottom + ItemSpacing;
            any = true;
        }

        pins.assign(spec.pins.size(), {});
        for (U64 i = 0; i < spec.pins.size(); ++i) {
            const auto& pin = spec.pins[i];
            const F32 top = any ? cursor : 0.0f;
            const F32 labelWidth = metrics.measure(MonoFont, pin.label, NodeFont * r) / r;
            widest = std::max(widest, labelWidth);
            const F32 x = pin.isInput ? 0.0f : std::max(0.0f, width - labelWidth);
            pins[i].label = {x, top, labelWidth, NodeFont};
            lastBottom = top + NodeFont;
            cursor = lastBottom + ItemSpacing;
            any = true;
        }

        hasContent = content && !content->empty();
        contentTop = hasContent ? (any ? lastBottom + content->leadingGap() : 0.0f) : lastBottom;
        F32 naturalHeight = lastBottom;
        fixedHeight = lastBottom;
        minimumHeight = lastBottom;
        contentHeight = 0.0f;
        F32 restoredHeight = naturalHeight;
        if (hasContent) {
            const auto measured = content->metrics(ctx, width * r);
            fixedHeight = contentTop + measured.fixed / r;
            minimumHeight = contentTop + measured.minimum / r;
            naturalHeight = contentTop + measured.natural / r;
            contentHeight = measured.natural / r;
            restoredHeight = fixedHeight + std::max(spec.minimumDimensions.y, measured.flexibleBasis / r);
        }

        if (HasAxis(spec.resize, ResizeAxes::Y) && spec.state != NodeState::Loading) {
            const F32 seeded = contentSize.y > 0.0f ? contentSize.y : restoredHeight;
            contentSize.y = std::max(seeded, effectiveMinimumHeight());
        }

        F32 height = naturalHeight;
        if (contentSize.y > 0.0f) {
            const F32 floor = (hasContent && content->hasFlexible()) ? minimumHeight : naturalHeight;
            height = std::max(contentSize.y, floor);
            if (hasContent) {
                contentHeight = std::max(0.0f, height - contentTop);
            }
        }
        if (spec.state == NodeState::Loading) {
            height = std::max(height, kLoadingMinHeight);
        }

        rect = {origin.x, origin.y, widest + 2.0f * kPadding, height + 2.0f * kPadding};

        for (U64 i = 0; i < pins.size(); ++i) {
            const auto& pin = spec.pins[i];
            const F32 cy = origin.y + kPadding + pins[i].label.y + NodeFont * 0.5f;
            pins[i].center = {pin.isInput ? rect.x : rect.right(), cy};
        }
        measured = Measured{
            .revision = layoutRevision(),
            .epoch = Access::layoutEpoch(),
            .fonts = ctx.render ? ctx.render->fontRevision() : 0,
            .ratio = ctx.pixelRatio,
            .origin = origin,
            .contentSize = contentSize,
        };
    }

    NodeEditor::Geometry geometry() const {
        return {
            .gridPosition = origin,
            .contentDimensions = {contentSize.x > 0.0f ? contentSize.x : contentWidth,
                                  contentSize.y > 0.0f ? contentSize.y : rect.height - 2.0f * kPadding},
        };
    }

    Rect chevronGridRect() const {
        return {origin.x + kPadding + chevronRect.x, origin.y + kPadding + chevronRect.y,
                chevronRect.width, chevronRect.height};
    }

    Rect reach(const Rect& nodePx, F32 r) const {
        const F32 side = kPinRadius * r + 2.0f;
        const F32 below = (kTimingOffset + static_cast<F32>(spec.timing.size()) * NodeFont) * r + 2.0f;
        const F32 right = spec.timing.empty() ? side : std::max(side, nodePx.width * 3.0f);
        return {nodePx.x - side, nodePx.y - 2.0f, nodePx.width + side + right, nodePx.height + 2.0f + below};
    }

    Rect gripHandle() const {
        const F32 size = NodeFont * 0.9f;
        return {rect.right() - size, rect.bottom() - size, size, size};
    }

    struct Frame {
        Extent2D<F32> origin;
        F32 r = 1.0f;
        F32 scaling = 1.0f;
        F32 fringe = 1.0f;
        Rect viewport;
        bool gripHovered = false;
        bool gripActive = false;
        bool chevronHovered = false;
        bool selected = false;

        bool operator==(const Frame&) const = default;
    };

    Extent2D<F32> toPx(const Frame& f, Extent2D<F32> grid) const {
        return {f.origin.x + grid.x * f.r, f.origin.y + grid.y * f.r};
    }

    Rect toPx(const Frame& f, const Rect& grid) const {
        const auto p = toPx(f, Extent2D<F32>{grid.x, grid.y});
        return {p.x, p.y, grid.width * f.r, grid.height * f.r};
    }

    void draw(const Context& ctx, const Frame& f) {
        const F32 r = f.r;
        const Rect nodePx = toPx(f, rect);
        bodyRect = nodePx;
        const Rect clipRect = f.viewport;
        const auto stateColor = ctx.color(StateColorKey(spec.state));
        const auto pinColor = spec.state == NodeState::Normal ? ctx.color("node_pin") : stateColor;
        const F32 halfBorder = kBorderThickness * 0.5f * r;
        const F32 half = halfBorder - 0.5f * r / f.scaling;

        chrome.update({
            .instances = {{
                .rect = {nodePx.x - half, nodePx.y - half, nodePx.width + 2.0f * half, nodePx.height + 2.0f * half},
                .backgroundColor = ctx.color("node_background"),
            }},
            .clip = clipRect,
            .cornerRadius = kCornerRounding * r + halfBorder,
            .borderWidth = kBorderThickness * r,
            .borderColor = f.selected ? Lighten(stateColor, kSelectedLighten, 1.0f) : stateColor,
        });

        const Extent2D<F32> gridOrigin = f.origin;
        std::vector<Polyline::Instance> shapeInstances;
        for (const auto& pin : pins) {
            shapeInstances.push_back({
                .points = Polyline::Circle(pin.center, kPinRadius, kPinSegments),
                .color = pinColor,
                .filled = true,
            });
        }
        std::vector<Polyline::Instance> gripInstances;
        if (spec.resize != ResizeAxes::None) {
            const auto gripBase = ctx.color(f.gripActive ? "resize_grip_active"
                                            : f.gripHovered ? "resize_grip_hovered"
                                                            : "resize_grip");
            const auto gripColor = Doubled(gripBase);
            const F32 margin = kPadding * 0.5f;
            const F32 size = NodeFont * 0.60f;
            const F32 step = size * 0.35f;
            const Extent2D<F32> corner = {rect.right() - margin, rect.bottom() - margin};
            for (int line = 0; line < 2; ++line) {
                const F32 length = size - static_cast<F32>(line) * step;
                gripInstances.push_back({
                    .points = {{corner.x - length, corner.y}, {corner.x, corner.y - length}},
                    .color = gripColor,
                    .thickness = 1.5f,
                });
            }
        }
        shapes.update({
            .instances = std::move(shapeInstances),
            .clip = clipRect,
            .origin = gridOrigin,
            .scale = r,
            .fringe = f.fringe,
        });
        grip.update({
            .instances = std::move(gripInstances),
            .clip = clipRect,
            .origin = gridOrigin,
            .scale = r,
            .fringe = f.fringe,
            .capacity = kGripCapacity,
        });

        const Extent2D<F32> top = toPx(f, Extent2D<F32>{origin.x + kPadding, origin.y + kPadding});
        std::vector<Label::Instance> headerInstances;
        std::vector<Label::Instance> textInstances;
        if (spec.header) {
            headerInstances.push_back({
                .rect = {top.x + titleX * r, top.y, std::max(0.0f, contentWidth - titleX) * r, kTitleFont * r},
                .str = displayTitle,
                .color = ctx.color("text_primary"),
                .fontSize = kTitleFont * r,
                .alignment = {0, 1},
            });
            const F32 contentOrigin = kTitleFont + 2.0f * kPadding;
            headerInstances.push_back({
                .rect = {top.x + subtitleX * r, top.y + (contentOrigin - kSubtitlePull) * r,
                         std::max(1.0f, rect.width - 2.0f * kPadding - subtitleX) * r, kSubtitleFont * r},
                .str = spec.subtitle,
                .color = ctx.color("text_secondary"),
                .fontSize = kSubtitleFont * r,
                .alignment = {0, 1},
            });
        }
        for (U64 i = 0; i < pins.size(); ++i) {
            const auto& label = pins[i].label;
            textInstances.push_back({
                .rect = {top.x + label.x * r, top.y + label.y * r, std::max(1.0f, label.width) * r, NodeFont * r},
                .str = spec.pins[i].label,
                .color = ctx.color("text_primary"),
                .fontSize = NodeFont * r,
                .alignment = {0, 1},
            });
        }
        header.update({
            .instances = std::move(headerInstances),
            .clip = clipRect,
            .fontName = HeaderFont,
            .maxCharacters = 128,
            .capacity = 2,
        });
        texts.update({
            .instances = std::move(textInstances),
            .clip = clipRect,
            .fontName = MonoFont,
            .maxCharacters = 128,
            .capacity = std::max<U64>(kTextCapacity, pins.size()),
        });

        std::vector<Label::Instance> iconInstances;
        if (chevronRect.width > 0.0f) {
            iconInstances.push_back({
                .rect = {top.x + chevronRect.x * r, top.y + chevronRect.y * r, chevronRect.width * r, chevronRect.height * r},
                .str = spec.collapsed ? ICON_FA_CHEVRON_RIGHT : ICON_FA_CHEVRON_DOWN,
                .color = ctx.color(f.chevronHovered ? "text_primary" : "text_secondary"),
                .fontSize = NodeFont * r,
                .alignment = {1, 1},
            });
        }
        if (skullRect.width > 0.0f) {
            iconInstances.push_back({
                .rect = {top.x + skullRect.x * r, top.y + skullRect.y * r, skullRect.width * r, skullRect.height * r},
                .str = ICON_FA_SKULL,
                .color = stateColor,
                .fontSize = NodeFont * r,
                .alignment = {1, 1},
            });
        }
        icons.update({
            .instances = std::move(iconInstances),
            .clip = clipRect,
            .fontName = IconFont,
            .maxCharacters = 4,
            .capacity = 2,
        });

        drawLoading(ctx, f, nodePx);

        std::vector<Label::Instance> timingInstances;
        for (U64 i = 0; i < spec.timing.size(); ++i) {
            const F32 y = nodePx.bottom() + kTimingOffset * r + static_cast<F32>(i) * NodeFont * r;
            timingInstances.push_back({
                .rect = {nodePx.x, y, std::max(1.0f, nodePx.width * 4.0f), NodeFont * r},
                .str = spec.timing[i],
                .color = ctx.color(i + 1 == spec.timing.size() ? "text_primary" : "text_secondary"),
                .fontSize = NodeFont * r,
                .alignment = {0, 1},
            });
        }
        timing.update({
            .instances = std::move(timingInstances),
            .clip = clipRect,
            .fontName = MonoFont,
            .maxCharacters = 128,
            .capacity = std::bit_ceil(std::max<U64>(kTimingCapacity, spec.timing.size())),
        });

        for (Component* part : {static_cast<Component*>(&chrome), static_cast<Component*>(&shapes),
                                static_cast<Component*>(&grip), static_cast<Component*>(&header),
                                static_cast<Component*>(&texts), static_cast<Component*>(&icons),
                                static_cast<Component*>(&timing)}) {
            layoutChild(ctx, *part, f.viewport);
        }
        if (loadingMounted) {
            for (Component* part : {static_cast<Component*>(&loadingBorder), static_cast<Component*>(&chase),
                                    static_cast<Component*>(&badge), static_cast<Component*>(&badgeText)}) {
                layoutChild(ctx, *part, f.viewport);
            }
        }

        if (hasContent) {
            const Rect contentRect = toPx(f, Rect{origin.x + kPadding, origin.y + kPadding + contentTop,
                                                  contentWidth, contentHeight});
            layoutChild(ctx, *content, contentRect, bodyRect);
        } else if (content) {
            layoutChild(ctx, *content, Rect{});
        }

    }

    I64 tooltipIndexAt(Extent2D<F32> gridPoint) const {
        const Extent2D<F32> top = {origin.x + kPadding, origin.y + kPadding};
        for (U64 i = 0; i < pins.size() && i < spec.pins.size(); ++i) {
            const Rect label = {top.x + pins[i].label.x, top.y + pins[i].label.y, pins[i].label.width, NodeFont};
            if (spec.pins[i].tooltip && label.contains(gridPoint.x, gridPoint.y)) {
                return static_cast<I64>(i);
            }
        }
        const Rect skull = {top.x + skullRect.x, top.y + skullRect.y, skullRect.width, skullRect.height};
        if (skullRect.width > 0.0f && spec.diagnosticTooltip && skull.contains(gridPoint.x, gridPoint.y)) {
            return static_cast<I64>(pins.size());
        }
        return -1;
    }

    std::optional<std::pair<Rect, Component*>> tooltipAt(Extent2D<F32> gridPoint) const {
        const Extent2D<F32> top = {origin.x + kPadding, origin.y + kPadding};
        for (U64 i = 0; i < pins.size() && i < spec.pins.size(); ++i) {
            const Rect label = {top.x + pins[i].label.x, top.y + pins[i].label.y, pins[i].label.width, NodeFont};
            if (spec.pins[i].tooltip && label.contains(gridPoint.x, gridPoint.y)) {
                return std::make_pair(label, spec.pins[i].tooltip());
            }
        }
        const Rect skull = {top.x + skullRect.x, top.y + skullRect.y, skullRect.width, skullRect.height};
        if (skullRect.width > 0.0f && spec.diagnosticTooltip && skull.contains(gridPoint.x, gridPoint.y)) {
            return std::make_pair(skull, spec.diagnosticTooltip());
        }
        return std::nullopt;
    }

    void drawLoading(const Context& ctx, const Frame& f, const Rect& nodePx) {
        const bool loading = spec.state == NodeState::Loading;
        const F32 r = f.r;
        const F32 unit = r / f.scaling;
        const auto pending = ctx.color("node_outline_pending");
        const auto highlight = Lighten(pending, 0.22f, 0.92f);

        if (!loading) {
            return;
        }
        requestFrame();

        const F32 thickness = kBorderThickness * r;
        const F32 inset = std::max(thickness * 0.5f, r);
        const Rect path = {nodePx.x + inset, nodePx.y + inset,
                           std::max(0.0f, nodePx.width - 2.0f * inset), std::max(0.0f, nodePx.height - 2.0f * inset)};
        const F32 radius = std::max(0.0f, kCornerRounding * r - inset);
        const F32 perimeter = RoundedRectPerimeter(path, radius);
        const F32 time = static_cast<F32>(ctx.time);
        const F32 pulse = 0.55f + std::sin(time * 7.0f) * 0.25f;
        const F32 halfStroke = 0.5f * unit;

        loadingBorder.update({
            .instances = {{
                .rect = {path.x - thickness * 0.5f + halfStroke, path.y - thickness * 0.5f + halfStroke,
                         path.width + thickness - 2.0f * halfStroke, path.height + thickness - 2.0f * halfStroke},
                .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f},
            }},
            .clip = f.viewport,
            .cornerRadius = radius + thickness * 0.5f,
            .borderWidth = thickness,
            .borderColor = WithAlpha(highlight, pulse * 0.45f),
        });

        std::vector<Polyline::Instance> segments;
        const F32 visibleThickness = thickness + 0.75f * r;
        const F32 glowThickness = visibleThickness + 4.0f * r;
        const F32 chaseLength = std::min(perimeter * 0.42f, 360.0f * r);
        if (perimeter > 0.0f && chaseLength > 0.0f && path.width > 0.0f && path.height > 0.0f) {
            const F32 step = std::max(2.0f * r, chaseLength / 88.0f);
            const F32 head = std::fmod(time * 320.0f * r, perimeter);
            for (F32 offset = 0.0f; offset < chaseLength; offset += step) {
                const F32 nextOffset = std::min(offset + step, chaseLength);
                const auto p0 = PointOnRoundedRect(path, radius, head - offset);
                const auto p1 = PointOnRoundedRect(path, radius, head - nextOffset);
                const F32 dx = p1.x - p0.x;
                const F32 dy = p1.y - p0.y;
                if (dx * dx + dy * dy > step * step * 4.0f) {
                    continue;
                }
                const F32 t = 1.0f - offset / chaseLength;
                const F32 alpha = t * t;
                segments.push_back({.points = {p0, p1}, .color = WithAlpha(pending, alpha * 0.45f),
                                    .thickness = glowThickness});
                segments.push_back({.points = {p0, p1}, .color = WithAlpha(pending, alpha * 0.95f),
                                    .thickness = visibleThickness});
                segments.push_back({.points = {p0, p1}, .color = WithAlpha(highlight, alpha * 0.85f),
                                    .thickness = thickness});
            }
        }
        chase.update({
            .instances = std::move(segments),
            .clip = f.viewport,
            .fringe = f.fringe,
            .capacity = kChaseCapacity,
        });

        const char* label = "LOADING";
        const F32 fontSize = BaseFont * r;
        const F32 textWidth = metrics.measure(BoldFont, label, fontSize);
        const F32 padX = 10.0f * r;
        const F32 padY = 5.0f * r;
        const Extent2D<F32> badgeSize = {textWidth + padX * 2.0f, fontSize + padY * 2.0f};
        const F32 clearance = 48.0f * r;
        const F32 margin = 8.0f * r;
        const F32 centered = nodePx.y + (nodePx.height - badgeSize.y) * 0.5f + 12.0f * r;
        const F32 maxY = nodePx.bottom() - badgeSize.y - margin;
        const F32 badgeY = maxY >= nodePx.y + clearance ? std::clamp(centered, nodePx.y + clearance, maxY) : centered;
        const Rect badgeRect = {nodePx.x + (nodePx.width - badgeSize.x) * 0.5f, badgeY, badgeSize.x, badgeSize.y};

        badge.update({
            .instances = {{.rect = badgeRect, .backgroundColor = WithAlpha(pending, 0.16f)}},
            .clip = f.viewport,
            .cornerRadius = 8.0f * r,
        });
        badgeText.update({
            .instances = {{
                .rect = {badgeRect.x + padX, badgeRect.y + padY, textWidth, fontSize},
                .str = label,
                .color = pending,
                .fontSize = fontSize,
                .alignment = {0, 1},
            }},
            .clip = f.viewport,
            .fontName = BoldFont,
            .maxCharacters = 16,
        });
    }

    Frame current;
    Rect bodyRect;

    void place(const Frame& next) {
        if (next != current) {
            current = next;
            invalidate(Dirty::Layout);
        }
    }

    void layout(const Context& ctx) override {
        draw(ctx, current);
    }

    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override {
        if (bodyRect.contains(point.x, point.y)) {
            return SurfaceCursor::Default;
        }
        return std::nullopt;
    }

    bool dispatch(const MouseEvent& event) {
        if (!hasContent || !content) {
            return false;
        }
        return mouseEventChildren(fromParent(event));
    }
};

struct GraphRoot : public Component {
    enum class Interaction : U8 {
        None,
        Pan,
        Node,
        Resize,
        LinkCreation,
        Content,
        Box,
    };

    NodeEditor::Config config;
    std::unordered_map<std::string, std::unique_ptr<NodeFrame>> frames;
    std::vector<std::string> depth;
    std::vector<std::string> selection;

    Polyline grid;
    Curves links;
    Box marquee;
    Curves wire;

    Extent2D<F32> pan = {0.0f, 0.0f};
    F32 ratio = 1.0f;
    F32 scaling = 1.0f;
    F32 fringe = 1.0f;
    F32 zoom = 1.0f;
    Rect viewport;
    std::optional<Extent2D<F32>> mousePx;
    bool tooltipFollowing = false;

    Interaction interaction = Interaction::None;
    bool middlePressed = false;
    std::string interactionNode;
    Extent2D<F32> dragStartMouse = {0.0f, 0.0f};
    Extent2D<F32> panStart = {0.0f, 0.0f};
    std::vector<std::pair<std::string, Extent2D<F32>>> dragOrigins;
    Extent2D<F32> boxStart = {0.0f, 0.0f};
    std::vector<std::string> boxBase;
    Extent2D<F32> resizeStartContent = {0.0f, 0.0f};
    NodeEditor::PinRef linkStart;
    bool chevronArmed = false;
    Extent2D<F32> chevronPressMouse = {0.0f, 0.0f};
    bool childOrderDirty = true;

    std::string hoveredNode;
    std::optional<NodeEditor::PinRef> hoveredPin;
    std::string hoveredLink;
    std::string resizeHoverNode;
    bool chevronHovered = false;
    std::string moveTarget;

    struct LinkGeometry {
        std::string id;
        Bezier curve;
        bool unresolved = false;
        NodeEditor::PinRef start;
        NodeEditor::PinRef end;
        NodeOverlay tooltip;
    };
    std::vector<LinkGeometry> linkGeometry;
    std::vector<Rect> nodeRects;

    GraphRoot() {
        setClipsChildren(true);
    }

    Extent2D<F32> toGrid(Extent2D<F32> px) const {
        const auto offset = panOffset();
        return {(px.x - viewport.x - offset.x) / ratio, (px.y - viewport.y - offset.y) / ratio};
    }

    Extent2D<F32> panOffset() const {
        return {std::round(pan.x * ratio), std::round(pan.y * ratio)};
    }

    Extent2D<F32> toPx(Extent2D<F32> gridPoint) const {
        return GridToPx(viewport, panOffset(), ratio, gridPoint);
    }

    NodeFrame* frameFor(const std::string& id) {
        const auto it = frames.find(id);
        return it == frames.end() ? nullptr : it->second.get();
    }

    const NodeFrame* frameFor(const std::string& id) const {
        const auto it = frames.find(id);
        return it == frames.end() ? nullptr : it->second.get();
    }

    const NodeFrame::PinLayout* pinLayout(const NodeEditor::PinRef& ref, NodeFrame** owner = nullptr) {
        auto* frame = frameFor(ref.nodeId);
        if (!frame) {
            return nullptr;
        }
        for (U64 i = 0; i < frame->spec.pins.size(); ++i) {
            const auto& pin = frame->spec.pins[i];
            if (pin.id == ref.pinId && pin.isInput == ref.isInput && i < frame->pins.size()) {
                if (owner) {
                    *owner = frame;
                }
                return &frame->pins[i];
            }
        }
        return nullptr;
    }

    void apply(NodeEditor::Config next) {
        std::unordered_set<std::string> present;
        for (auto& node : next.nodes) {
            present.insert(node.id);
            auto& frame = frames[node.id];
            if (!frame) {
                frame = std::make_unique<NodeFrame>();
                depth.push_back(node.id);
                childOrderDirty = true;
            }
            frame->apply(std::move(node));
        }
        for (auto it = frames.begin(); it != frames.end();) {
            if (!present.contains(it->first)) {
                const std::string id = it->first;
                it = frames.erase(it);
                depth.erase(std::remove(depth.begin(), depth.end(), id), depth.end());
                selection.erase(std::remove(selection.begin(), selection.end(), id), selection.end());
                boxBase.erase(std::remove(boxBase.begin(), boxBase.end(), id), boxBase.end());
                if (interactionNode == id) {
                    interaction = Interaction::None;
                    interactionNode.clear();
                }
                childOrderDirty = true;
            } else {
                ++it;
            }
        }
        if (linkStart.nodeId.size() && !frames.contains(linkStart.nodeId) &&
            interaction == Interaction::LinkCreation) {
            interaction = Interaction::None;
        }
        next.nodes.clear();
        if (next != config || childOrderDirty) {
            invalidate(Dirty::Layout);
        }
        config = std::move(next);
        rebuildOrder();
    }

    void rebuildOrder() {
        if (!childOrderDirty) {
            return;
        }
        childOrderDirty = false;
        std::vector<Component*> order = {&grid, &links};
        for (const auto& id : depth) {
            if (auto* frame = frameFor(id)) {
                order.push_back(frame);
            }
        }
        order.push_back(&marquee);
        order.push_back(&wire);
        setChildren(order);
    }

    void raise(const std::vector<std::string>& ids) {
        bool changed = false;
        for (const auto& id : ids) {
            const auto it = std::find(depth.begin(), depth.end(), id);
            if (it == depth.end() || it + 1 == depth.end()) {
                continue;
            }
            depth.erase(it);
            depth.push_back(id);
            changed = true;
        }
        if (changed) {
            childOrderDirty = true;
            rebuildOrder();
        }
    }

    bool selected(const std::string& id) const {
        return std::find(selection.begin(), selection.end(), id) != selection.end();
    }

    std::string topmostNodeAt(Extent2D<F32> gridPoint) const {
        for (auto it = depth.rbegin(); it != depth.rend(); ++it) {
            const auto found = frames.find(*it);
            if (found != frames.end() && found->second->rect.contains(gridPoint.x, gridPoint.y)) {
                return *it;
            }
        }
        return {};
    }

    bool pinOccluded(const std::string& nodeId, Extent2D<F32> point) const {
        bool above = false;
        for (const auto& id : depth) {
            if (id == nodeId) {
                above = true;
                continue;
            }
            if (!above) {
                continue;
            }
            const auto found = frames.find(id);
            if (found != frames.end() && found->second->rect.contains(point.x, point.y)) {
                return true;
            }
        }
        return false;
    }

    std::optional<NodeEditor::PinRef> pinAt(Extent2D<F32> mouse) const {
        std::optional<NodeEditor::PinRef> found;
        F32 bestPin = (kPinHoverRadius / scaling) * (kPinHoverRadius / scaling);
        for (const auto& id : depth) {
            const auto* frame = frameFor(id);
            for (U64 i = 0; i < frame->pins.size() && i < frame->spec.pins.size(); ++i) {
                const auto& center = frame->pins[i].center;
                const F32 dx = center.x - mouse.x;
                const F32 dy = center.y - mouse.y;
                const F32 distance = dx * dx + dy * dy;
                if (distance < bestPin && !pinOccluded(id, center)) {
                    bestPin = distance;
                    found = NodeEditor::PinRef{
                        .nodeId = id,
                        .pinId = frame->spec.pins[i].id,
                        .isInput = frame->spec.pins[i].isInput,
                    };
                }
            }
        }
        return found;
    }

    std::optional<SurfaceCursor> cursorOverChildren(const Extent2D<F32>& point) const override {
        if (!viewport.contains(point.x, point.y)) {
            return std::nullopt;
        }
        if (interaction != Interaction::None && interaction != Interaction::LinkCreation) {
            return std::nullopt;
        }
        const auto mouse = toGrid(point);
        const std::string topmost = topmostNodeAt(mouse);
        if (topmost.empty()) {
            return std::nullopt;
        }
        const auto* frame = frameFor(topmost);
        if (frame->spec.resize != ResizeAxes::None && frame->gripHandle().contains(mouse.x, mouse.y)) {
            return SurfaceCursor::ResizeNWSE;
        }
        if (frame->chevronRect.width > 0.0f && frame->chevronGridRect().contains(mouse.x, mouse.y) &&
            !pinAt(mouse).has_value()) {
            return SurfaceCursor::Hand;
        }
        return std::nullopt;
    }

    void resolveHover() {
        hoveredNode.clear();
        hoveredPin.reset();
        hoveredLink.clear();
        resizeHoverNode.clear();
        chevronHovered = false;

        if (!mousePx.has_value() || !viewport.contains(mousePx->x, mousePx->y)) {
            return;
        }
        if (interaction != Interaction::None && interaction != Interaction::LinkCreation) {
            return;
        }
        const auto mouse = toGrid(*mousePx);

        hoveredPin = pinAt(mouse);

        const std::string topmost = topmostNodeAt(mouse);
        if (!topmost.empty()) {
            const auto* frame = frameFor(topmost);
            if (frame->spec.resize != ResizeAxes::None && frame->gripHandle().contains(mouse.x, mouse.y)) {
                resizeHoverNode = topmost;
            }
        }

        if (!hoveredPin.has_value()) {
            hoveredNode = topmost;
            if (!hoveredNode.empty()) {
                const auto* frame = frameFor(hoveredNode);
                chevronHovered = frame->chevronRect.width > 0.0f && frame->chevronGridRect().contains(mouse.x, mouse.y);
            }
        }

        if (hoveredNode.empty()) {
            F32 bestLink = std::numeric_limits<F32>::max();
            const F32 hoverDistance = kLinkHoverDistance / scaling;
            for (const auto& link : linkGeometry) {
                if (hoveredPin.has_value()) {
                    if (link.start == *hoveredPin || link.end == *hoveredPin) {
                        hoveredLink = link.id;
                        break;
                    }
                    continue;
                }
                if (!BezierBounds(link.curve, hoverDistance).contains(mouse.x, mouse.y)) {
                    continue;
                }
                const F32 distance = DistanceToBezier(link.curve, mouse);
                if (distance < hoverDistance && distance < bestLink) {
                    bestLink = distance;
                    hoveredLink = link.id;
                }
            }
        }
    }

    bool shouldSnap(const NodeEditor::PinRef& target) const {
        if (target.nodeId == linkStart.nodeId || target.isInput == linkStart.isInput) {
            return false;
        }
        for (const auto& link : config.links) {
            if ((link.start == linkStart && link.end == target) || (link.start == target && link.end == linkStart)) {
                return false;
            }
        }
        return true;
    }

    void zoomTo(F32 next) {
        next = std::clamp(next, kZoomSteps.front(), kZoomSteps.back());
        if (next == zoom) {
            return;
        }
        const F32 nextRatio = ratio / zoom * next;
        if (viewport.width > 0.0f && viewport.height > 0.0f) {
            const Extent2D<F32> center = {viewport.x + viewport.width * 0.5f, viewport.y + viewport.height * 0.5f};
            const Extent2D<F32> anchor = toGrid(center);
            pan = {(center.x - viewport.x) / nextRatio - anchor.x, (center.y - viewport.y) / nextRatio - anchor.y};
        }
        ratio = nextRatio;
        zoom = next;
        invalidate(Dirty::Layout);
    }

    void zoomStep(int direction) {
        if (direction > 0) {
            for (const F32 step : kZoomSteps) {
                if (step > zoom + 1e-3f) {
                    zoomTo(step);
                    return;
                }
            }
        } else {
            for (auto it = kZoomSteps.rbegin(); it != kZoomSteps.rend(); ++it) {
                if (*it < zoom - 1e-3f) {
                    zoomTo(*it);
                    return;
                }
            }
        }
    }

    void autoPan(const Context& ctx) {
        if (!mousePx.has_value()) {
            return;
        }
        const bool panning = interaction == Interaction::LinkCreation || interaction == Interaction::Node ||
                             interaction == Interaction::Box;
        if (!panning || viewport.contains(mousePx->x, mousePx->y)) {
            return;
        }
        const Extent2D<F32> center = {viewport.x + viewport.width * 0.5f, viewport.y + viewport.height * 0.5f};
        Extent2D<F32> direction = {center.x - mousePx->x, center.y - mousePx->y};
        const F32 length = Length(direction);
        if (length <= 0.0f) {
            return;
        }
        direction = {direction.x / length, direction.y / length};
        requestFrame();
        const F32 delta = ctx.deltaTime * kAutoPanSpeed / scaling;
        pan.x += direction.x * delta;
        pan.y += direction.y * delta;
    }

    void updateDrags() {
        if (!mousePx.has_value()) {
            return;
        }
        if (interaction == Interaction::Pan) {
            pan = {panStart.x + (mousePx->x - dragStartMouse.x) / ratio,
                   panStart.y + (mousePx->y - dragStartMouse.y) / ratio};
            return;
        }
        const auto mouse = toGrid(*mousePx);
        if (interaction == Interaction::Node) {
            const Extent2D<F32> delta = {mouse.x - dragStartMouse.x, mouse.y - dragStartMouse.y};
            for (const auto& [id, origin] : dragOrigins) {
                if (auto* frame = frameFor(id)) {
                    frame->origin = {origin.x + delta.x, origin.y + delta.y};
                }
            }
        } else if (interaction == Interaction::Box) {
            selectInBox(mouse);
        } else if (interaction == Interaction::Resize) {
            if (auto* frame = frameFor(interactionNode)) {
                const Extent2D<F32> delta = {mouse.x - dragStartMouse.x, mouse.y - dragStartMouse.y};
                if (HasAxis(frame->spec.resize, ResizeAxes::X)) {
                    frame->contentSize.x = std::max(frame->spec.minimumDimensions.x, resizeStartContent.x + delta.x);
                }
                if (HasAxis(frame->spec.resize, ResizeAxes::Y)) {
                    frame->contentSize.y = std::max(frame->effectiveMinimumHeight(), resizeStartContent.y + delta.y);
                }
                requestCursor(SurfaceCursor::ResizeNWSE);
            }
        }
    }

    void layout(const Context& ctx) override {
        viewport = frame();
        ratio = std::max(1e-3f, ctx.pixelRatio) * zoom;
        scaling = std::max(1e-3f, ScalingFactor(ctx));
        fringe = std::max(1e-3f, ctx.pixelRatio) / scaling;

        autoPan(ctx);
        updateDrags();

        Context measureCtx = ctx;
        measureCtx.pixelRatio = ratio;
        measureCtx.fontSize = NodeFont;
        measureCtx.backdropColorKey = "node_background";
        for (const auto& id : depth) {
            frameFor(id)->measureLayout(measureCtx);
        }

        linkGeometry.clear();
        for (const auto& link : config.links) {
            const auto* start = pinLayout(link.start);
            const auto* end = pinLayout(link.end);
            if (!start || !end) {
                continue;
            }
            linkGeometry.push_back({
                .id = link.id,
                .curve = LinkBezier(start->center, end->center, link.start.isInput, kLinkSegmentsPerLength),
                .unresolved = link.unresolved,
                .start = link.start,
                .end = link.end,
                .tooltip = link.tooltip,
            });
        }

        resolveHover();

        drawGrid(ctx);
        drawLinks(ctx);

        nodeRects.clear();
        nodeRects.reserve(depth.size());
        for (const auto& id : depth) {
            const auto* node = frameFor(id);
            const auto p = toPx(Extent2D<F32>{node->rect.x, node->rect.y});
            nodeRects.push_back({p.x, p.y, node->rect.width * ratio, node->rect.height * ratio});
        }

        const Extent2D<F32> offset = panOffset();
        const Rect localViewport = {viewport.x - offset.x, viewport.y - offset.y, viewport.width, viewport.height};

        Context nodeCtx = ctx.translated(offset);
        nodeCtx.pixelRatio = ratio;
        nodeCtx.fontSize = NodeFont;
        nodeCtx.backdropColorKey = "node_background";
        for (U64 i = 0; i < depth.size(); ++i) {
            auto* frame = frameFor(depth[i]);
            const bool onscreen = depth[i] == interactionNode ||
                                  (interaction == Interaction::LinkCreation && depth[i] == linkStart.nodeId) ||
                                  Overlaps(frame->reach(nodeRects[i], ratio), viewport);
            if (frame->visible() != onscreen) {
                frame->setVisible(onscreen);
            }
            if (!onscreen) {
                continue;
            }
            frame->place({
                .origin = {viewport.x, viewport.y},
                .r = ratio,
                .scaling = scaling,
                .fringe = fringe,
                .viewport = localViewport,
                .gripHovered = resizeHoverNode == depth[i],
                .gripActive = interaction == Interaction::Resize && interactionNode == depth[i],
                .chevronHovered = chevronHovered && hoveredNode == depth[i],
                .selected = selected(depth[i]),
            });
            layoutChild(nodeCtx, *frame, localViewport);
        }

        drawInteraction(ctx);
        requestEditorCursor();
        emitGeometry();

        requestTooltips(ctx);
    }

    void requestTooltips(const Context& ctx) {
        tooltipFollowing = false;
        if (interaction != Interaction::None || !mousePx.has_value() || !viewport.contains(mousePx->x, mousePx->y)) {
            return;
        }
        const auto mouse = toGrid(*mousePx);
        if (const auto* frame = frameFor(topmostNodeAt(mouse))) {
            if (const auto found = frame->tooltipAt(mouse)) {
                const auto corner = toPx(Extent2D<F32>{found->first.x, found->first.y});
                const Rect anchor = {corner.x, corner.y, found->first.width * ratio, found->first.height * ratio};
                requestTooltip(ctx, anchor, found->second);
            }
            return;
        }
        if (hoveredLink.empty() || hoveredPin.has_value()) {
            return;
        }
        for (const auto& link : linkGeometry) {
            if (link.id == hoveredLink && !link.unresolved && link.tooltip) {
                requestTooltip(ctx, {mousePx->x, mousePx->y, 1.0f, 1.0f}, link.tooltip());
                break;
            }
        }
    }

    void requestTooltip(const Context& ctx, const Rect& anchor, Component* content) {
        tooltipFollowing = true;
        if (!content) {
            requestFrame();
            return;
        }
        RequestPopup(ctx, TooltipRequest(ctx, anchor, content));
    }

    void drawGrid(const Context& ctx) {
        const F32 spacing = kGridSpacing;
        const F32 width = viewport.width / ratio;
        const F32 height = viewport.height / ratio;
        const F32 offset = 0.5f / scaling;
        const auto color = ctx.color("grid_line");
        const F32 firstX = (std::floor(-pan.x / spacing) - 1.0f) * spacing;
        const F32 lastX = (std::ceil((width - pan.x) / spacing) + 1.0f) * spacing;
        const F32 firstY = (std::floor(-pan.y / spacing) - 1.0f) * spacing;
        const F32 lastY = (std::ceil((height - pan.y) / spacing) + 1.0f) * spacing;
        std::vector<Polyline::Instance> lines;
        for (F32 x = firstX; x <= lastX; x += spacing) {
            lines.push_back({
                .points = {{x + offset, firstY + offset}, {x + offset, lastY + offset}},
                .color = color,
                .thickness = 1.0f / scaling,
            });
        }
        for (F32 y = firstY; y <= lastY; y += spacing) {
            lines.push_back({
                .points = {{firstX + offset, y + offset}, {lastX + offset, y + offset}},
                .color = color,
                .thickness = 1.0f / scaling,
            });
        }
        grid.update({
            .instances = std::move(lines),
            .clip = viewport,
            .origin = {viewport.x + panOffset().x, viewport.y + panOffset().y},
            .scale = ratio,
            .fringe = fringe,
            .capacity = kGridCapacity,
        });
        layoutChild(ctx, grid, viewport);
    }

    Curves::Instance curveInstance(const Bezier& curve, ColorRGBA<F32> color) const {
        return {
            .p0 = curve.p0,
            .p1 = curve.p1,
            .p2 = curve.p2,
            .p3 = curve.p3,
            .color = color,
            .thickness = kLinkThickness * ratio,
            .segments = curve.segments,
        };
    }

    Extent2D<F32> curveOrigin() const {
        const auto offset = panOffset();
        return {viewport.x + offset.x, viewport.y + offset.y};
    }

    void drawLinks(const Context& ctx) {
        const auto linkColor = ctx.color("node_link");
        const auto pendingColor = ctx.color("node_outline_pending");
        std::vector<Curves::Instance> wires;
        wires.reserve(linkGeometry.size());
        for (const auto& link : linkGeometry) {
            wires.push_back(curveInstance(link.curve, link.unresolved ? pendingColor : linkColor));
        }
        links.update({
            .instances = std::move(wires),
            .origin = curveOrigin(),
            .scale = ratio,
            .clip = viewport,
            .fringe = fringe,
            .capacity = kLinkCapacity,
            .maxSegments = kLinkMaxSegments,
        });
        layoutChild(ctx, links, viewport);
    }

    void drawMarquee(const Context& ctx) {
        std::vector<Box::Instance> instances;
        if (interaction == Interaction::Box && mousePx.has_value()) {
            const Rect box = boxRect(toGrid(*mousePx));
            const auto corner = toPx(Extent2D<F32>{box.x, box.y});
            instances.push_back({
                .rect = {corner.x, corner.y, box.width * ratio, box.height * ratio},
                .backgroundColor = WithAlpha(ctx.color("text_primary"), kMarqueeFill),
            });
        }
        marquee.update({
            .instances = std::move(instances),
            .clip = viewport,
            .borderWidth = Hairline(ctx),
            .borderColor = WithAlpha(ctx.color("text_primary"), kMarqueeBorder),
        });
        layoutChild(ctx, marquee, viewport);
    }

    void drawInteraction(const Context& ctx) {
        drawMarquee(ctx);
        std::vector<Curves::Instance> instances;
        if (interaction == Interaction::LinkCreation && mousePx.has_value()) {
            if (const auto* start = pinLayout(linkStart)) {
                Extent2D<F32> end = toGrid(*mousePx);
                if (hoveredPin.has_value() && shouldSnap(*hoveredPin)) {
                    if (const auto* target = pinLayout(*hoveredPin)) {
                        end = target->center;
                    }
                }
                auto curve = LinkBezier(start->center, end, linkStart.isInput, kLinkSegmentsPerLength);
                curve.segments = std::min<U64>(curve.segments, kLinkMaxSegments);
                instances.push_back(curveInstance(curve, ctx.color("node_link")));
            }
        }
        wire.update({
            .instances = std::move(instances),
            .origin = curveOrigin(),
            .scale = ratio,
            .clip = viewport,
            .fringe = fringe,
            .capacity = 1,
            .maxSegments = kLinkMaxSegments,
        });
        layoutChild(ctx, wire, viewport);
    }

    void emitGeometry() {
        for (const auto& id : depth) {
            auto* frame = frameFor(id);
            if (!frame->spec.onGeometryChange) {
                continue;
            }
            const auto geometry = frame->geometry();
            bool changed = !frame->lastGeometry.has_value();
            if (!changed) {
                const auto& last = *frame->lastGeometry;
                const auto near = [](F32 a, F32 b) {
                    return std::abs(a - b) <= 0.5f;
                };
                changed = !(near(last.gridPosition.x, geometry.gridPosition.x) &&
                            near(last.gridPosition.y, geometry.gridPosition.y) &&
                            near(last.contentDimensions.x, geometry.contentDimensions.x) &&
                            near(last.contentDimensions.y, geometry.contentDimensions.y));
            }
            if (changed) {
                frame->lastGeometry = geometry;
                frame->spec.onGeometryChange(geometry);
            }
        }
    }

    const NodeEditor::Link* linkSpec(const std::string& id) const {
        for (const auto& link : config.links) {
            if (link.id == id) {
                return &link;
            }
        }
        return nullptr;
    }

    void beginNodeDrag(const std::string& id) {
        if (!selected(id)) {
            selection = {id};
        }
        interaction = Interaction::Node;
        interactionNode = id;
        dragStartMouse = toGrid(*mousePx);
        dragOrigins.clear();
        for (const auto& selectedId : selection) {
            if (const auto* frame = frameFor(selectedId)) {
                dragOrigins.emplace_back(selectedId, frame->origin);
            }
        }
        raise(selection);
    }

    void toggleSelection(const std::string& id) {
        if (selected(id)) {
            selection.erase(std::remove(selection.begin(), selection.end(), id), selection.end());
            return;
        }
        selection.push_back(id);
        beginNodeDrag(id);
    }

    Rect boxRect(Extent2D<F32> corner) const {
        return {std::min(boxStart.x, corner.x), std::min(boxStart.y, corner.y), std::abs(corner.x - boxStart.x),
                std::abs(corner.y - boxStart.y)};
    }

    void selectInBox(Extent2D<F32> corner) {
        const Rect box = boxRect(corner);
        selection = boxBase;
        for (const auto& id : depth) {
            const auto* frame = frameFor(id);
            if (frame && !selected(id) && Overlaps(frame->rect, box)) {
                selection.push_back(id);
            }
        }
    }

    void requestEditorCursor() {
        if (interaction == Interaction::Resize) {
            requestCursor(SurfaceCursor::ResizeNWSE);
        }
    }

    struct HoverKey {
        std::string node;
        std::optional<NodeEditor::PinRef> pin;
        std::string link;
        std::string resize;
        bool chevron = false;
        std::string tooltip;

        bool operator==(const HoverKey&) const = default;
    };

    HoverKey hoverKey() {
        resolveHover();
        HoverKey key = {
            .node = hoveredNode,
            .pin = hoveredPin,
            .link = hoveredLink,
            .resize = resizeHoverNode,
            .chevron = chevronHovered,
        };
        if (interaction == Interaction::None && mousePx.has_value() && viewport.contains(mousePx->x, mousePx->y)) {
            const auto mouse = toGrid(*mousePx);
            const std::string topmost = topmostNodeAt(mouse);
            if (const auto* frame = frameFor(topmost)) {
                key.tooltip = topmost + ":" + std::to_string(frame->tooltipIndexAt(mouse));
            }
        }
        return key;
    }

    bool mouseEvent(const MouseEvent& event) override {
        const HoverKey before = hoverKey();
        const auto previous = mousePx;
        const bool handled = dispatchMouse(event);
        if (event.type != MouseEventType::Move || hoverKey() != before ||
            ((interaction != Interaction::None || tooltipFollowing) && mousePx != previous)) {
            invalidate(Dirty::Layout);
        }
        return handled;
    }

    bool dispatchMouse(const MouseEvent& event) {
        if (event.type == MouseEventType::Leave || event.type == MouseEventType::Cancel) {
            mousePx.reset();
        } else {
            mousePx = Extent2D<F32>{event.position.x, event.position.y};
        }

        switch (event.type) {
            case MouseEventType::Move: {
                if (interaction == Interaction::Content) {
                    if (auto* frame = frameFor(interactionNode)) {
                        frame->dispatch(event);
                    }
                    return true;
                }
                if (interaction == Interaction::None) {
                    const std::string topmost = topmostNodeAt(toGrid(*mousePx));
                    if (topmost != moveTarget) {
                        if (auto* previous = frameFor(moveTarget)) {
                            MouseEvent leave = event;
                            leave.type = MouseEventType::Leave;
                            previous->dispatch(leave);
                        }
                        moveTarget = topmost;
                    }
                    if (auto* frame = frameFor(topmost)) {
                        frame->dispatch(event);
                    }
                    requestEditorCursor();
                    return false;
                }
                requestEditorCursor();
                return true;
            }
            case MouseEventType::Scroll: {
                const auto mouse = toGrid(*mousePx);
                const std::string topmost = topmostNodeAt(mouse);
                if (!topmost.empty()) {
                    return frameFor(topmost)->dispatch(event);
                }
                return false;
            }
            case MouseEventType::Leave: {
                moveTarget.clear();
                for (const auto& id : depth) {
                    frameFor(id)->dispatch(event);
                }
                return false;
            }
            case MouseEventType::Release: {
                return release(event);
            }
            case MouseEventType::Cancel: {
                cancelInteraction();
                for (const auto& id : depth) {
                    frameFor(id)->dispatch(event);
                }
                return false;
            }
            case MouseEventType::Click: {
                if (event.button == MouseButton::Right) {
                    return rightClick();
                }
                if (event.button == MouseButton::Middle) {
                    middlePressed = true;
                    return true;
                }
                return leftClick(event);
            }
            default:
                return false;
        }
    }

    bool rightClick() {
        const auto mouse = toGrid(*mousePx);
        const std::string topmost = topmostNodeAt(mouse);
        if (!topmost.empty()) {
            if (auto* frame = frameFor(topmost); frame && frame->spec.onContextMenu) {
                frame->spec.onContextMenu();
            }
            return true;
        }
        if (config.onEditorContextMenu) {
            config.onEditorContextMenu(mouse);
        }
        return true;
    }

    bool leftClick(const MouseEvent& event) {
        if (interaction != Interaction::None) {
            return true;
        }
        resolveHover();
        const auto mouse = toGrid(*mousePx);
        const bool doubleClick = event.clicks >= 2;

        if (!resizeHoverNode.empty()) {
            auto* frame = frameFor(resizeHoverNode);
            interaction = Interaction::Resize;
            interactionNode = resizeHoverNode;
            dragStartMouse = mouse;
            resizeStartContent = {frame->rect.width - 2.0f * kPadding, frame->rect.height - 2.0f * kPadding};
            return true;
        }

        if (!hoveredLink.empty()) {
            const auto* link = linkSpec(hoveredLink);
            if (link && hoveredPin.has_value()) {
                const auto* frame = frameFor(hoveredPin->nodeId);
                bool detach = false;
                if (frame) {
                    for (const auto& pin : frame->spec.pins) {
                        if (pin.id == hoveredPin->pinId && pin.isInput == hoveredPin->isInput) {
                            detach = pin.enableDetach;
                        }
                    }
                }
                if (detach) {
                    if (config.onLinkDestroyed) {
                        config.onLinkDestroyed(link->id);
                    }
                } else {
                    interaction = Interaction::LinkCreation;
                    linkStart = *hoveredPin;
                }
            }
            if (!hoveredPin.has_value()) {
                selection.clear();
            }
            if (doubleClick && hoveredNode.empty() && config.onEditorDoubleClick) {
                config.onEditorDoubleClick(mouse);
            }
            return true;
        }

        if (hoveredPin.has_value()) {
            interaction = Interaction::LinkCreation;
            linkStart = *hoveredPin;
            return true;
        }

        if (!hoveredNode.empty()) {
            auto* frame = frameFor(hoveredNode);
            if (frame->dispatch(event)) {
                interaction = Interaction::Content;
                interactionNode = hoveredNode;
                selection = {hoveredNode};
                raise(selection);
                return true;
            }
            if (event.modifiers.shift || event.modifiers.control || event.modifiers.super) {
                toggleSelection(hoveredNode);
                return true;
            }
            chevronArmed = chevronHovered;
            chevronPressMouse = *mousePx;
            beginNodeDrag(hoveredNode);
            return true;
        }

        if (doubleClick && config.onEditorDoubleClick) {
            config.onEditorDoubleClick(mouse);
        }
        if (event.modifiers.shift) {
            interaction = Interaction::Box;
            boxStart = mouse;
            boxBase = selection;
            return true;
        }
        selection.clear();
        interaction = Interaction::Pan;
        dragStartMouse = *mousePx;
        panStart = pan;
        return true;
    }

    void cancelInteraction() {
        middlePressed = false;
        if (interaction == Interaction::Content) {
            if (auto* frame = frameFor(interactionNode)) {
                MouseEvent cancel{};
                cancel.type = MouseEventType::Cancel;
                frame->dispatch(cancel);
            }
        }
        interaction = Interaction::None;
        interactionNode.clear();
        dragOrigins.clear();
        chevronArmed = false;
    }

    bool release(const MouseEvent& event) {
        if (event.button == MouseButton::Middle) {
            return std::exchange(middlePressed, false);
        }
        if (event.button != MouseButton::Left) {
            for (const auto& id : depth) {
                frameFor(id)->dispatch(event);
            }
            return false;
        }

        switch (interaction) {
            case Interaction::Content:
                if (auto* frame = frameFor(interactionNode)) {
                    frame->dispatch(event);
                }
                break;
            case Interaction::LinkCreation: {
                resolveHover();
                if (hoveredPin.has_value() && shouldSnap(*hoveredPin) && config.onLinkCreated) {
                    config.onLinkCreated(linkStart, *hoveredPin);
                }
                break;
            }
            case Interaction::Box:
                selectInBox(toGrid(*mousePx));
                break;
            case Interaction::Node: {
                auto* frame = frameFor(interactionNode);
                if (chevronArmed && mousePx.has_value() && frame && frame->chevronRect.width > 0.0f) {
                    const F32 dx = std::abs(mousePx->x - chevronPressMouse.x) / ratio;
                    const F32 dy = std::abs(mousePx->y - chevronPressMouse.y) / ratio;
                    const auto mouse = toGrid(*mousePx);
                    if (frame->chevronGridRect().contains(mouse.x, mouse.y) && dx < kChevronDragTolerance &&
                        dy < kChevronDragTolerance && frame->spec.onToggleCollapse) {
                        frame->spec.onToggleCollapse();
                    }
                }
                break;
            }
            default:
                break;
        }
        chevronArmed = false;
        if (interaction != Interaction::Content) {
            for (const auto& id : depth) {
                frameFor(id)->dispatch(event);
            }
        }
        interaction = Interaction::None;
        interactionNode.clear();
        dragOrigins.clear();
        return true;
    }
};

struct NodeEditor::Impl {
    GraphRoot root;
    Box background;
    Polyline corners;
    Box border;
    Rect bounds;
    std::optional<Extent2D<F32>> reportedCenter;
};

NodeEditor::NodeEditor() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->background);
    addChild(impl->root);
    addChild(impl->corners);
    addChild(impl->border);
}

NodeEditor::~NodeEditor() = default;

void NodeEditor::update(Config config) {
    impl->root.apply(std::move(config));
}

Extent2D<F32> NodeEditor::measure(const Context&, Extent2D<F32> available) {
    return {FillLength(available.x), FillLength(available.y)};
}

void NodeEditor::layout(const Context& ctx) {
    auto& root = impl->root;
    const auto& config = root.config;
    const Rect bounds = frame();
    const F32 radius = kEditorRounding * ctx.pixelRatio;

    impl->background.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color("background")}},
        .cornerRadius = radius,
    });
    layoutChild(ctx, impl->background, bounds);
    layoutChild(ctx, root, bounds);
    const auto backdrop = ctx.color(ctx.backdropColorKey.empty() ? "background" : ctx.backdropColorKey);
    impl->corners.update({
        .instances = Polyline::CornerMasks(bounds, radius, backdrop, backdrop),
        .clip = bounds,
        .capacity = 256,
    });
    layoutChild(ctx, impl->corners, bounds);
    impl->border.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f}}},
        .cornerRadius = radius,
        .borderWidth = ctx.pixelRatio,
        .borderColor = ctx.color("border"),
        .strokeFringe = true,
    });
    layoutChild(ctx, impl->border, bounds);

    if (config.onViewportGridCenterChange && !bounds.empty()) {
        const auto center = root.toGrid({bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f});
        if (impl->reportedCenter != center) {
            impl->reportedCenter = center;
            config.onViewportGridCenterChange(center);
        }
    }

    impl->bounds = bounds;
}

bool NodeEditor::keyEvent(const KeyEvent& event) {
    auto& root = impl->root;
    const auto& config = root.config;
    if (event.type != KeyEventType::Press || textEditorFocused() ||
        !(event.modifiers.control || event.modifiers.super)) {
        return false;
    }
    const bool initial = !event.repeat;
    switch (event.key) {
        case KeyCode::C:
            if (!initial || !config.onCopyShortcut) {
                return false;
            }
            config.onCopyShortcut(root.selection);
            return true;
        case KeyCode::V: {
            if (!initial || !config.pasteEnabled || !config.onPasteShortcut) {
                return false;
            }
            const Rect& bounds = impl->bounds;
            const auto& mouse = root.mousePx;
            const bool pointed = mouse.has_value() && bounds.contains(mouse->x, mouse->y);
            const Extent2D<F32> point = pointed ? *mouse
                                                : Extent2D<F32>{bounds.x + bounds.width * 0.5f,
                                                                bounds.y + bounds.height * 0.5f};
            config.onPasteShortcut(root.toGrid(point));
            return true;
        }
        case KeyCode::Equal:
        case KeyCode::KeypadAdd:
            root.zoomStep(1);
            return true;
        case KeyCode::Minus:
        case KeyCode::KeypadSubtract:
            root.zoomStep(-1);
            return true;
        case KeyCode::Digit0:
        case KeyCode::Keypad0:
            if (!initial) {
                return false;
            }
            root.zoomTo(1.0f);
            return true;
        default:
            return false;
    }
}

bool NodeEditor::focusesOnClick() const {
    return true;
}

F32 NodeEditor::zoom() const {
    return impl->root.zoom;
}

bool NodeEditor::canZoom(int direction) const {
    const F32 zoom = impl->root.zoom;
    return direction > 0 ? zoom < kZoomSteps.back() - 1e-3f : zoom > kZoomSteps.front() + 1e-3f;
}

void NodeEditor::zoomStep(int direction) {
    impl->root.zoomStep(direction);
}

void NodeEditor::zoomReset() {
    impl->root.zoomTo(1.0f);
}

}  // namespace Jetstream::Sakura
