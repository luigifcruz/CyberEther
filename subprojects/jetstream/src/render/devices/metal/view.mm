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

    auto nativeWindow = static_cast<NSWindow*>(nativeWindowHolder);
    auto swapchain = static_cast<CAMetalLayer*>(swapchainHolder);

    swapchain.device = (id<MTLDevice>)device;
    swapchain.pixelFormat = MTLPixelFormatBGRA8Unorm;

    nativeWindow.contentView.layer = swapchain;
    nativeWindow.contentView.wantsLayer = YES;
}

View::~View() {
    [static_cast<CAMetalLayer*>(swapchainHolder) release];
}

CA::MetalDrawable* View::draw() {
    auto swapchain = static_cast<CAMetalLayer*>(swapchainHolder);

    @autoreleasepool {
        int width, height;
        glfwGetFramebufferSize(glfwWindow, &width, &height);
        swapchain.drawableSize = CGSizeMake(width, height);
    }

    id<CAMetalDrawable> drawable = [swapchain nextDrawable];

    return (__bridge CA::MetalDrawable*)drawable;
}

}  // namespace Render

}  // namespace Jetstream
