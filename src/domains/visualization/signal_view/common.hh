#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_COMMON_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_COMMON_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <jetstream/domains/visualization/signal_view/module.hh>
#include <jetstream/memory/tensor.hh>
#include <jetstream/surface.hh>
#include <jetstream/render/base.hh>
#include <jetstream/render/components/axis.hh>
#include <jetstream/render/components/shapes.hh>
#include <jetstream/render/components/text.hh>

namespace Jetstream::Modules {

namespace detail {

constexpr F32 kSignalViewLineThickness = 1.0f;

inline bool SignalViewHasLineplot(const std::string& mode) {
    return mode == "lineplot" || mode == "lineplot_waterfall";
}

inline bool SignalViewHasWaterfall3D(const std::string& mode) {
    return mode == "waterfall_3d";
}

inline bool SignalViewHasWaterfall(const std::string& mode) {
    return mode == "waterfall" || mode == "lineplot_waterfall" ||
           SignalViewHasWaterfall3D(mode);
}

inline bool HostReadable(const Tensor& tensor) {
    return (static_cast<U8>(tensor.buffer().location()) &
            static_cast<U8>(Location::Host)) != 0 && tensor.data();
}

inline std::string LabelUnit(const std::string& label) {
    const auto open = label.rfind('(');
    const auto close = label.rfind(')');
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
        return {};
    }
    return label.substr(open + 1, close - open - 1);
}

constexpr U64 MaxMarkers = 16;
constexpr U64 MarkerSpans = MaxMarkers - 1;

inline std::string FormatFrequencySpan(const F64 hertz) {
    const F64 magnitude = std::abs(hertz);
    const char* unit = "Hz";
    F64 value = magnitude;
    int decimals = 1;
    if (magnitude >= 1.0e6) {
        unit = "MHz";
        value = magnitude / 1.0e6;
        decimals = 3;
    } else if (magnitude >= 1.0e3) {
        unit = "kHz";
        value = magnitude / 1.0e3;
        decimals = 3;
    }
    std::string text = jst::fmt::format("{:.{}f}", value, decimals);
    while (text.size() > 2 && text.back() == '0' && text[text.size() - 2] != '.') {
        text.pop_back();
    }
    return jst::fmt::format("{} {}", text, unit);
}


constexpr F32 MinSplitRatio = 0.1f;
constexpr F32 MaxSplitRatio = 0.9f;

struct SignalViewPanelLayout {
    Render::ScissorRect plot;
    Render::ScissorRect line;
    Render::ScissorRect waterfall;
    F32 lineFraction = 0.5f;
};

inline SignalViewPanelLayout CalculateSignalViewPanels(const Extent2D<F32>& padding,
                                                       const Extent2D<U64>& size,
                                                       F32 ratio) {
    SignalViewPanelLayout layout;
    const F32 x = std::clamp(padding.x, 0.0f, 1.0f);
    const F32 y = std::clamp(padding.y, 0.0f, 1.0f);
    layout.plot.x = static_cast<U32>((1.0f - x) * 0.5f * size.x);
    layout.plot.y = static_cast<U32>((1.0f - y) * 0.5f * size.y);
    layout.plot.width = static_cast<U32>(x * size.x);
    layout.plot.height = static_cast<U32>(y * size.y);
    layout.line = layout.plot;
    layout.waterfall = layout.plot;
    ratio = std::isfinite(ratio) ? std::clamp(ratio, MinSplitRatio, MaxSplitRatio) : 0.5f;
    layout.lineFraction = ratio;
    if (layout.plot.height >= 2) {
        layout.line.height = std::clamp(static_cast<U32>(std::lround(layout.plot.height * ratio)),
                                        U32{1}, layout.plot.height - 1);
        layout.lineFraction = static_cast<F32>(layout.line.height) / layout.plot.height;
    } else {
        layout.line.height = layout.plot.height;
    }
    layout.waterfall.y += layout.line.height;
    layout.waterfall.height -= layout.line.height;
    return layout;
}

// Gesture state is a preview, not the applied module configuration. Release
// emits one edit; leaving/cancelling discards the preview. Coordinates outside
// the surface remain valid while the frontend holds mouse capture.
struct SignalViewSplitInteraction {
    F32 ratio = 0.5f;
    F32 grabOffset = 0.0f;
    bool dragging = false;

    bool hovered(const Extent2D<F32>& position,
                 const SignalViewPanelLayout& layout,
                 const Extent2D<U64>& size,
                 const F32 scale,
                 const bool enabled) const {
        const F32 x = position.x * size.x;
        const F32 y = position.y * size.y;
        return enabled && layout.plot.height >= 2 &&
               std::isfinite(x) && std::isfinite(y) &&
               x >= layout.plot.x && x <= layout.plot.x + layout.plot.width &&
               std::abs(y - layout.waterfall.y) <= 6.0f * scale;
    }

    bool process(const MouseEvent& event,
                 const SignalViewPanelLayout& layout,
                 const Extent2D<U64>& size,
                 F32 scale,
                 bool enabled,
                 bool& commit) {
        commit = false;
        if (event.type == MouseEventType::Leave) {
            const bool consumed = dragging;
            dragging = false;
            return consumed;
        }
        const F32 x = event.position.x * size.x;
        const F32 y = event.position.y * size.y;
        if (dragging) {
            if ((event.type == MouseEventType::Move ||
                 (event.type == MouseEventType::Release && event.button == MouseButton::Left)) &&
                layout.plot.height >= 2 && std::isfinite(y)) {
                ratio = std::clamp((y - layout.plot.y - grabOffset) / layout.plot.height,
                                   MinSplitRatio, MaxSplitRatio);
            }
            if (event.type == MouseEventType::Release && event.button == MouseButton::Left) {
                dragging = false;
                commit = true;
            }
            return true;
        }
        if (event.type == MouseEventType::Click && event.button == MouseButton::Left &&
            hovered(event.position, layout, size, scale, enabled)) {
            dragging = true;
            grabOffset = y - (layout.plot.y + ratio * layout.plot.height);
            return true;
        }
        return false;
    }
};

}  // namespace detail


struct WaterfallAveragingPlan {
    U64 rowCount = 0;
    U64 pendingRows = 0;
};

inline WaterfallAveragingPlan PlanWaterfallAveraging(const U64 pendingRows,
                                                    const U64 incomingRows,
                                                    const U64 averaging) {
    const U64 neededRows = averaging - pendingRows;
    if (incomingRows < neededRows) {
        return {.rowCount = 0, .pendingRows = pendingRows + incomingRows};
    }
    const U64 remainingRows = incomingRows - neededRows;
    return {
        .rowCount = 1 + remainingRows / averaging,
        .pendingRows = remainingRows % averaging,
    };
}

struct WaterfallWritePlan {
    U64 sourceRow = 0;
    U64 destinationRow = 0;
    U64 rowCount = 0;
};

inline WaterfallWritePlan PlanWaterfallWrite(const U64 writeIndex,
                                             const U64 numberOfBatches,
                                             const U64 height) {
    const U64 retainedRows = std::min(numberOfBatches, height);
    const U64 sourceRow = numberOfBatches - retainedRows;
    return {
        .sourceRow = sourceRow,
        .destinationRow = (writeIndex + (sourceRow % height)) % height,
        .rowCount = retainedRows,
    };
}

struct WaterfallDirtyPlan {
    U64 startRow = 0;
    U64 firstRowCount = 0;
    U64 secondRowCount = 0;
};

struct WaterfallHistory {
    U64 writeIndex = 0;
    U64 dirtyRows = 0;

    void advance(const U64 numberOfBatches, const U64 height) {
        writeIndex = (writeIndex + (numberOfBatches % height)) % height;
        dirtyRows += std::min(numberOfBatches, height - dirtyRows);
    }

    WaterfallDirtyPlan dirtyPlan(const U64 height) const {
        const U64 startRow = (writeIndex + height - dirtyRows) % height;
        const U64 firstRowCount = std::min(dirtyRows, height - startRow);
        return {
            .startRow = startRow,
            .firstRowCount = firstRowCount,
            .secondRowCount = dirtyRows - firstRowCount,
        };
    }

    void clearDirty() {
        dirtyRows = 0;
    }
};

struct WaterfallFrame {
    const F32* bins = nullptr;
    U64 writeIndex = 0;
    WaterfallDirtyPlan dirty;
};

struct SignalViewFrequency {
    bool valid = false;
    F32 center = 0.0f;
    F32 sampleRate = 0.0f;

    bool operator==(const SignalViewFrequency&) const = default;
};

SignalViewFrequency SignalViewFrequencyOf(const Tensor& input);

struct SignalViewLineplot;
struct SignalViewWaterfall;

struct SignalViewCanvas {
    struct Context {
        const SignalView& config;
        SignalViewFrequency frequency;
        U64 numberOfElements = 0;
        SignalViewLineplot* lineplot = nullptr;
        SignalViewWaterfall* waterfall = nullptr;
    };

    struct Edits {
        bool splitEnabled = false;
        std::function<bool()> pending;
        std::function<bool(const std::string&)> enabled;
        std::function<Result(const Parser::Map&)> request;
    };

    struct CursorState {
        bool inside = false;
        Extent2D<F32> position = {0.0f, 0.0f};
        bool visible = false;
        bool marker = false;
        bool overMarker = false;
        F32 point = 0.0f;
        Extent2D<F32> plot = {0.0f, 0.0f};
    };

    struct TagBounds {
        bool active = false;
        Extent2D<F32> center = {0.0f, 0.0f};
        Extent2D<F32> halfSize = {0.0f, 0.0f};
    };

    struct MarkerDrag {
        std::optional<U64> index;
        bool moved = false;
        Extent2D<F32> origin = {0.0f, 0.0f};
    };

    void reset(const SignalView& config);
    Result create(const std::shared_ptr<Render::Window>& window, const Context& context);
    Result destroy(const std::shared_ptr<Render::Window>& window);

    Result processSurfaceEvents(std::vector<SurfaceEvent>&& events);
    SurfaceCursor processInputEvents(std::vector<InputEvent>&& events,
                                     const Context& context,
                                     const Edits& edits);
    void resize();
    void updateState(const Context& context);
    Result present(const Context& context);

    SurfaceInteractionState interaction;
    detail::SignalViewSplitInteraction splitter;
    CursorState cursor;
    bool displayHeld = false;
    Extent2D<F32> pixelSize;

    std::vector<F32> markerPositions;
    bool updateMarkersFlag = false;
    std::array<TagBounds, detail::MaxMarkers> tagBounds;
    std::array<bool, detail::MaxMarkers> pinned{};
    MarkerDrag markerDrag;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Surface> renderSurface;
    std::shared_ptr<Render::Components::Axis> axis;
    std::shared_ptr<Render::Components::Text> text;
    std::shared_ptr<Render::Components::Shapes> cursorShapes;
    std::shared_ptr<Render::Components::Text> cursorText;
    std::shared_ptr<Render::Components::Shapes> markerShapes;
    std::shared_ptr<Render::Components::Shapes> markerTagShapes;
    std::shared_ptr<Render::Components::Shapes> markerSpanShapes;
    std::shared_ptr<Render::Components::Shapes> markerTableShapes;
    std::shared_ptr<Render::Components::Text> markerText;
    std::shared_ptr<Render::Components::Text> markerBadgeText;
    std::shared_ptr<Render::Components::Text> markerTagText;

 private:
    void updateLabels(const Context& context);
    Result updateCursor(const Context& context);
    Result updateMarkers(const Context& context);
    F32 viewTranslation() const;
    std::optional<F32> cursorPoint(const Context& context) const;
    F32 projectPointX(F32 xPoint) const;
    std::optional<F32> displayedAmplitude(const Context& context, F32 xPoint) const;
    F32 amplitudeToNdc(const Context& context, F32 yPoint) const;
    std::string formatPointX(const Context& context, F32 xPoint) const;
    std::string formatSpanX(const Context& context, F32 delta) const;
    std::string formatAmplitude(const Context& context, F32 yPoint) const;
    void syncMarkers(const Context& context, const Edits& edits);
    void applyPins(const std::vector<U64>& pins);
    std::vector<U64> pinnedIndices() const;
    std::optional<U64> tagAt(const Extent2D<F32>& position) const;
    std::optional<U64> markerAt(const Context& context, const Extent2D<F32>& position) const;
    bool insidePlot(const Extent2D<F32>& position) const;
    F32 pointAtX(F32 x) const;
    void toggleMarker(const Context& context, const Edits& edits);
    void clearMarkers(const Context& context, const Edits& edits);
    void commitMarkers(const Context& context, const Edits& edits);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_COMMON_HH
