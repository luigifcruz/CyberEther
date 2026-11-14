#ifndef JETSTREAM_VIEWPORT_GLFW_WAYLAND_POINTER_HH
#define JETSTREAM_VIEWPORT_GLFW_WAYLAND_POINTER_HH

#include <functional>
#include <memory>

#include "../../tools/touch_gesture.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

class WaylandPointer {
 public:
    struct Scroll {
        Render::InputQueue::Wheel event;
        Extent2D<F32> position;
    };

    using Touch = TouchGesture::Event;

    WaylandPointer(GLFWwindow* window, std::function<void(const Scroll&)> onScroll,
                   std::function<void(const Touch&)> onTouch = {});
    ~WaylandPointer();

    WaylandPointer(const WaylandPointer&) = delete;
    WaylandPointer& operator=(const WaylandPointer&) = delete;

    bool available() const;
    void focus(bool focused);
    void poll();

 private:
    struct State;
    std::unique_ptr<State> state;
};

}  // namespace Jetstream::Viewport

#endif
