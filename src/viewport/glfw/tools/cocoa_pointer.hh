#ifndef JETSTREAM_VIEWPORT_GLFW_COCOA_POINTER_HH
#define JETSTREAM_VIEWPORT_GLFW_COCOA_POINTER_HH

#include <functional>
#include <memory>
#include <optional>

#include "jetstream/surface.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

class CocoaPointer {
 public:
    struct Scroll {
        ScrollPhase phase = ScrollPhase::None;
        KeyModifiers modifiers;
    };

    CocoaPointer(GLFWwindow* window, std::function<void()> onIdle, std::function<void(KeyModifiers)> onModifiers);
    ~CocoaPointer();

    CocoaPointer(const CocoaPointer&) = delete;
    CocoaPointer& operator=(const CocoaPointer&) = delete;

    std::optional<Scroll> take();

 private:
    struct State;

    std::shared_ptr<State> state;
};

}  // namespace Jetstream::Viewport

#endif
