#ifndef JETSTREAM_VIEWPORT_GLFW_WINDOWS_POINTER_HH
#define JETSTREAM_VIEWPORT_GLFW_WINDOWS_POINTER_HH

#include <functional>
#include <memory>

#include "jetstream/surface.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

class WindowsPointer {
 public:
    struct Scroll {
        Extent2D<F32> delta = {0.0f, 0.0f};
        ScrollPhase phase = ScrollPhase::None;
        KeyModifiers modifiers;
    };

    WindowsPointer(GLFWwindow* window, std::function<void(const Scroll&)> onScroll);
    ~WindowsPointer();

    WindowsPointer(const WindowsPointer&) = delete;
    WindowsPointer& operator=(const WindowsPointer&) = delete;

    void poll();
    bool available() const;
    static KeyModifiers Modifiers();

 private:
    struct State;
    std::unique_ptr<State> state;
};

}  // namespace Jetstream::Viewport

#endif
