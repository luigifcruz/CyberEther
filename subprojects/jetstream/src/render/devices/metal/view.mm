#include "jetstream/render/metal/view.hh"
#include "jetstream/render/tools/imgui_impl_metal.h"

#include <GLFW/glfw3native.h>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Foundation/Foundation.h>

namespace Jetstream {

namespace Render {

View::View(MTL::Device* device, GLFWwindow* window) : glfwWindow(window) {
    swapchainHolder = [CAMetalLayer layer];
    nativeWindowHolder = glfwGetCocoaWindow(glfwWindow);

    auto nativeWindow = (__bridge NSWindow*)nativeWindowHolder;
    auto swapchain = (__bridge CAMetalLayer*)swapchainHolder;

    swapchain.device = (__bridge id<MTLDevice>)device;
    swapchain.pixelFormat = MTLPixelFormatBGRA8Unorm;

    nativeWindow.contentView.layer = swapchain;
    nativeWindow.contentView.wantsLayer = YES;
}

View::~View() {
    [(__bridge CAMetalLayer*)swapchainHolder release];
}

CA::MetalDrawable* View::draw() {
    auto swapchain = (__bridge CAMetalLayer*)swapchainHolder;

    int width, height;
    glfwGetFramebufferSize(glfwWindow, &width, &height);
    swapchain.drawableSize = CGSizeMake(width, height);

    return (__bridge CA::MetalDrawable*)[swapchain nextDrawable];
}

}  // namespace Render

}  // namespace Jetstream
