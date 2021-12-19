#include "render/metal/window.hpp"
#include "render/tools/imgui_impl_metal.h"

#include <GLFW/glfw3native.h>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Foundation/Foundation.h>

namespace Render {

MetalWindow::MetalWindow(MTL::Device* device, GLFWwindow* window) : glfwWindow(window) {
    swapchainHolder = [CAMetalLayer layer];
    nativeWindowHolder = glfwGetCocoaWindow(glfwWindow);

    auto nativeWindow = static_cast<NSWindow*>(nativeWindowHolder);
    auto swapchain = static_cast<CAMetalLayer*>(swapchainHolder);

    swapchain.device = (id<MTLDevice>)device;
    swapchain.pixelFormat = MTLPixelFormatBGRA8Unorm;

    nativeWindow.contentView.layer = swapchain;
    nativeWindow.contentView.wantsLayer = YES;
}

MetalWindow::~MetalWindow() {
    [static_cast<CAMetalLayer*>(swapchainHolder) release];
}

CA::MetalDrawable* MetalWindow::draw() {
    auto swapchain = static_cast<CAMetalLayer*>(swapchainHolder);

    int width, height;
    glfwGetFramebufferSize(glfwWindow, &width, &height);
    swapchain.drawableSize = CGSizeMake(width, height);
    id<CAMetalDrawable> drawable = [swapchain nextDrawable];

    return (__bridge CA::MetalDrawable*)drawable;
}

}  // namespace Render
