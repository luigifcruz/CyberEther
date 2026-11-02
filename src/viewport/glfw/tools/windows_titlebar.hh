#ifndef JETSTREAM_VIEWPORT_GLFW_WINDOWS_TITLEBAR_HH
#define JETSTREAM_VIEWPORT_GLFW_WINDOWS_TITLEBAR_HH

#include <functional>
#include <memory>

#include "jetstream/render/types.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

class WindowsTitlebar {
 public:
    WindowsTitlebar(GLFWwindow* window, std::function<void()> releaseLeftButton);
    ~WindowsTitlebar();

    WindowsTitlebar(const WindowsTitlebar&) = delete;
    WindowsTitlebar& operator=(const WindowsTitlebar&) = delete;

    void poll();
    Render::TitlebarChrome chrome() const;
    void beginDrag();
    void zoom();
    static bool SetAppearance(GLFWwindow* window, bool dark);

 private:
    struct State;
    std::unique_ptr<State> state;
};

}  // namespace Jetstream::Viewport

#endif
