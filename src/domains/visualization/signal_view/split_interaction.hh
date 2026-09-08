#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_SPLIT_INTERACTION_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_SPLIT_INTERACTION_HH

#include <algorithm>
#include <cmath>

#include <jetstream/render/base/program.hh>
#include <jetstream/surface.hh>

namespace Jetstream::Modules::detail {

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

}  // namespace Jetstream::Modules::detail

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_SPLIT_INTERACTION_HH
