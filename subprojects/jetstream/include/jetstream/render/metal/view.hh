#ifndef JETSTREAM_RENDER_METAL_VIEW_HH
#define JETSTREAM_RENDER_METAL_VIEW_HH

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>

#include "jetstream/backend/devices/metal/base.hh"

namespace Jetstream {

namespace Render {

class View {
 public:
    explicit View(MTL::Device* device, GLFWwindow* glfwWindow);
    ~View();

    CA::MetalDrawable* draw();

 private:
    void* swapchainHolder = nullptr;
    void* nativeWindowHolder = nullptr;
    GLFWwindow* glfwWindow = nullptr;
};

}  // namespace Render

}  // namespace Jetstream

#endif

