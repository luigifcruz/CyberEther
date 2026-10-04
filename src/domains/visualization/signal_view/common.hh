#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_COMMON_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_COMMON_HH

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
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
        if (enabled && event.type == MouseEventType::Click &&
            event.button == MouseButton::Left && layout.plot.height >= 2 &&
            std::isfinite(x) && std::isfinite(y) &&
            x >= layout.plot.x && x <= layout.plot.x + layout.plot.width &&
            std::abs(y - layout.waterfall.y) <= 6.0f * scale) {
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

    struct SplitEdits {
        F32 ratio = 0.5f;
        bool enabled = false;
        std::function<bool()> pending;
        std::function<Result(F32)> request;
    };

    struct CursorState {
        bool inside = false;
        Extent2D<F32> position = {0.0f, 0.0f};
        bool visible = false;
        bool marker = false;
        Extent2D<F32> plot = {0.0f, 0.0f};
    };

    Result create(const std::shared_ptr<Render::Window>& window, const Context& context);
    Result destroy(const std::shared_ptr<Render::Window>& window);

    Result processSurfaceEvents(std::vector<SurfaceEvent>&& events);
    void processInputEvents(std::vector<InputEvent>&& events, const SplitEdits& edits);
    void resize();
    void updateState(const Context& context);
    Result present(const Context& context);

    SurfaceInteractionState interaction;
    detail::SignalViewSplitInteraction splitter;
    CursorState cursor;
    bool displayHeld = false;
    Extent2D<F32> pixelSize;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Surface> renderSurface;
    std::shared_ptr<Render::Components::Axis> axis;
    std::shared_ptr<Render::Components::Text> text;
    std::shared_ptr<Render::Components::Shapes> cursorShapes;
    std::shared_ptr<Render::Components::Text> cursorText;

 private:
    void updateLabels(const Context& context);
    Result updateCursor(const Context& context);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_COMMON_HH
