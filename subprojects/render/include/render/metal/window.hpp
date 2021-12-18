#ifndef RENDER_METALWINDOW_H
#define RENDER_METALWINDOW_H

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <Foundation/Foundation.hpp>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>

namespace Render {

class MetalWindow {
 public:
   explicit MetalWindow(MTL::Device* device, GLFWwindow* glfwWindow);
   ~MetalWindow();

   CA::MetalDrawable* draw();

 private:
    void* swapchainHolder;
    void* nativeWindowHolder;

    GLFWwindow* glfwWindow;
};

}  // namespace Render

#endif
