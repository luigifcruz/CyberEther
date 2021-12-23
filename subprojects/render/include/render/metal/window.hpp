#ifndef RENDER_METALWINDOW_H
#define RENDER_METALWINDOW_H

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <Foundation/Foundation.hpp>

namespace Render {

class MetalWindow {
 public:
    explicit MetalWindow(MTL::Device* device, GLFWwindow* glfwWindow);
    ~MetalWindow();

    CA::MetalDrawable* draw();

 private:
    void* swapchainHolder = nullptr;
    void* nativeWindowHolder = nullptr;

    GLFWwindow* glfwWindow = nullptr;
};

}  // namespace Render

#endif
