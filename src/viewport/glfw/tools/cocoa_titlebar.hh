#ifndef JETSTREAM_VIEWPORT_GLFW_COCOA_TITLEBAR_HH
#define JETSTREAM_VIEWPORT_GLFW_COCOA_TITLEBAR_HH

#include <functional>
#include <memory>

#include "jetstream/render/types.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

class CocoaTitlebar {
 public:
    CocoaTitlebar(GLFWwindow* window, std::function<void()> releaseLeftButton);
    ~CocoaTitlebar();

    CocoaTitlebar(const CocoaTitlebar&) = delete;
    CocoaTitlebar& operator=(const CocoaTitlebar&) = delete;

    void poll();

    Render::TitlebarChrome chrome() const;
    void beginDrag();
    void zoom();

    static bool SetAppearance(GLFWwindow* window, bool dark);

 private:
    struct State;

    std::shared_ptr<State> state;
};

}  // namespace Jetstream::Viewport

#endif
