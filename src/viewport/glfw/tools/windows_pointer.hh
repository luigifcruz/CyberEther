#ifndef JETSTREAM_VIEWPORT_GLFW_WINDOWS_POINTER_HH
#define JETSTREAM_VIEWPORT_GLFW_WINDOWS_POINTER_HH

#include <functional>
#include <memory>

#include "jetstream/surface.hh"
#include "../../tools/touch_gesture.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

class WindowsPointer {
 public:
    struct Scroll {
        Extent2D<F32> delta = {0.0f, 0.0f};
        ScrollPhase phase = ScrollPhase::None;
        KeyModifiers modifiers;
    };

    using Touch = TouchGesture::Event;

    WindowsPointer(GLFWwindow* window, std::function<void(const Scroll&)> onScroll,
                   std::function<void(const Touch&)> onTouch = {},
                   std::function<bool(const Extent2D<F32>&)> captionHit = {});
    ~WindowsPointer();

    WindowsPointer(const WindowsPointer&) = delete;
    WindowsPointer& operator=(const WindowsPointer&) = delete;

    bool available() const;
    void focus(bool focused);
    void poll();
    static KeyModifiers Modifiers();

 private:
    struct State;
    std::unique_ptr<State> state;
};

}  // namespace Jetstream::Viewport

#endif
