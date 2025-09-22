#ifndef JETSTREAM_SURFACE_HH
#define JETSTREAM_SURFACE_HH

#include <string>
#include <vector>
#include <memory>

#include "jetstream/types.hh"
#include "jetstream/render/base/texture.hh"

namespace Jetstream {

//
// SurfaceManifest
//

struct JETSTREAM_API SurfaceManifest {
    std::string id;
    Extent2D<U64> size;
    std::shared_ptr<const Render::Texture> surface;
    bool forwardMouseEvents = false;
};

//
// MouseEvent
//

enum class JETSTREAM_API MouseButton : U8 {
    Left = 0,
    Right = 1,
};

enum class JETSTREAM_API MouseEventType : U8 {
    Click = 0,
    Release = 1,
    Move = 2,
    Scroll = 3,
    Enter = 4,
    Leave = 5,
};

struct JETSTREAM_API MouseEvent {
    MouseEventType type;
    MouseButton button;
    Extent2D<F32> position;
    Extent2D<F32> scroll;
};

//
// SurfaceEvent
//

enum class JETSTREAM_API SurfaceEventType : U8 {
    Resize = 0,
};

struct JETSTREAM_API SurfaceEvent {
    SurfaceEventType type;
    Extent2D<U64> size;
    F32 scale = 1.0f;
    ColorRGBA<F32> backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};
};

//
// EventBuffer
//

struct JETSTREAM_API EventBuffer {
    std::vector<MouseEvent> mouseEvents;
    std::vector<SurfaceEvent> surfaceEvents;

    void pushMouse(const MouseEvent& event) {
        mouseEvents.push_back(event);
    }

    void pushSurface(const SurfaceEvent& event) {
        surfaceEvents.push_back(event);
    }

    std::vector<MouseEvent> consumeMouseEvents() {
        return std::move(mouseEvents);
    }

    std::vector<SurfaceEvent> consumeSurfaceEvents() {
        return std::move(surfaceEvents);
    }
};

//
// SurfaceInteractionState
//

struct JETSTREAM_API SurfaceInteractionState {
    F32 zoom = 1.0f;
    F32 offset = 0.0f;
    F32 scale = 1.0f;
    Extent2D<U64> viewSize = {512, 512};
    ColorRGBA<F32> backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};

    Extent2D<F32> cursorNormalized = {0.0f, 0.0f};

    bool dragging = false;
    F32 dragAnchor = 0.0f;

    bool viewChanged = false;
    bool cursorMoved = false;
};

struct JETSTREAM_API SurfaceInteractionConfig {
    F32 zoomSpeed = 0.1f;
    F32 minZoom = 1.0f;
    F32 maxZoom = 10.0f;
    bool enableZoom = true;
    bool enablePan = true;
    bool enableCursor = true;
};

SurfaceInteractionState ProcessSurfaceInteraction(SurfaceInteractionState state,
                                                  std::vector<SurfaceEvent>&& surfaceEvents,
                                                  std::vector<MouseEvent>&& mouseEvents,
                                                  const SurfaceInteractionConfig& config = {});

}  // namespace Jetstream

#endif  // JETSTREAM_SURFACE_HH
