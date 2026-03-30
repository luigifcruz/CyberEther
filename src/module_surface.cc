#include <cmath>

#include <jetstream/detail/module_surface_impl.hh>
#include <jetstream/surface.hh>

namespace Jetstream {

SurfaceInteractionState ProcessSurfaceInteraction(SurfaceInteractionState state,
                                                   std::vector<SurfaceEvent>&& surfaceEvents,
                                                   std::vector<MouseEvent>&& mouseEvents,
                                                   const SurfaceInteractionConfig& config) {
    // Reset change flags.

    state.viewChanged = false;
    state.cursorMoved = false;

    // Process surface events.

    for (const auto& event : surfaceEvents) {
        if (event.type == SurfaceEventType::Resize) {
            const bool validSize = event.size.x > 0 && event.size.y > 0;
            const bool sizeChanged = validSize && event.size != state.viewSize;
            const bool scaleChanged = std::abs(event.scale - state.scale) > 1e-6f;

            if (sizeChanged || scaleChanged) {
                if (sizeChanged) {
                    state.viewSize = event.size;
                }
                state.scale = event.scale;
                state.viewChanged = true;
            }
            state.backgroundColor = event.backgroundColor;
        }
    }

    // Calculate max offset.

    const F32 maxOffset = 0.5f * (1.0f - (1.0f / state.zoom));

    // Process mouse events.

    for (const auto& event : mouseEvents) {
        switch (event.type) {
            case MouseEventType::Scroll: {
                if (!config.enableZoom) break;

                const F32 oldZoom = state.zoom;
                state.zoom = std::clamp(
                    state.zoom + event.scroll.y * config.zoomSpeed,
                    config.minZoom,
                    config.maxZoom
                );

                if (state.zoom != oldZoom) {
                    const F32 cursor = event.position.x;
                    const F32 zoomDelta = (1.0f / state.zoom) - (1.0f / oldZoom);
                    const F32 newOffset = state.offset - (cursor - 0.5f) * zoomDelta;

                    const F32 maxOffset = 0.5f * (1.0f - (1.0f / state.zoom));
                    state.offset = std::clamp(newOffset, -maxOffset, maxOffset);
                    state.viewChanged = true;
                }
                break;
            }
            case MouseEventType::Click: {
                if (event.button == MouseButton::Left && config.enablePan) {
                    state.dragging = true;
                    state.dragAnchor = (event.position.x / state.zoom) + state.offset;
                }
                if (event.button == MouseButton::Right && config.enableZoom) {
                    state.zoom = config.minZoom;
                    state.offset = 0.0f;
                    state.viewChanged = true;
                }
                break;
            }
            case MouseEventType::Release: {
                if (event.button == MouseButton::Left) {
                    state.dragging = false;
                }
                break;
            }
            case MouseEventType::Move: {
                if (config.enableCursor) {
                    state.cursorNormalized = event.position;
                    state.cursorMoved = true;
                }

                if (state.dragging && config.enablePan) {
                    const F32 newOffset = state.dragAnchor - (event.position.x / state.zoom);
                    state.offset = std::clamp(newOffset, -maxOffset, maxOffset);
                    state.viewChanged = true;
                }
                break;
            }
            case MouseEventType::Enter: {
                break;
            }
            case MouseEventType::Leave: {
                state.dragging = false;
                break;
            }
        }
    }

    return state;
}

Module::Surface::Surface() {
    impl = std::make_shared<Impl>();
}

Module::Surface::~Surface() {
    impl.reset();
}

const std::vector<SurfaceManifest>& Module::Surface::manifests() const {
    return impl->manifests;
}

void Module::Surface::pushMouseEvent(const MouseEvent& event) {
    impl->eventBuffer.pushMouse(event);
}

void Module::Surface::pushSurfaceEvent(const SurfaceEvent& event) {
    impl->eventBuffer.pushSurface(event);
}

}  // namespace Jetstream
