#ifndef JETSTREAM_VIEWPORT_GLFW_INPUT_HH
#define JETSTREAM_VIEWPORT_GLFW_INPUT_HH

#include <memory>
#include <string>

#include "jetstream/surface.hh"
#include "jetstream/viewport/adapters/generic.hh"

struct GLFWwindow;

namespace Jetstream::Viewport {

class GlfwInput final : public Generic::Platform {
 public:
    GlfwInput(GLFWwindow* window, Generic& viewport);
    ~GlfwInput() override;

    GlfwInput(const GlfwInput&) = delete;
    GlfwInput& operator=(const GlfwInput&) = delete;

    F64 time() const override;
    void poll() override;
    void setCursor(SurfaceCursor cursor) override;
    std::string clipboardText() const override;
    void setClipboardText(const std::string& text) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Viewport

#endif  // JETSTREAM_VIEWPORT_GLFW_INPUT_HH
