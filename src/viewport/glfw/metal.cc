#include <csignal>

#include "jetstream/viewport/platforms/glfw/metal.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "tools/imgui_impl_glfw.h"

namespace Jetstream::Viewport {

static void PrintGLFWError(int, const char* description) {
    JST_FATAL("[Metal] GLFW error: {}", description);
}

static bool keepRunningFlag;

using Implementation = GLFW<Device::Metal>;

Implementation::GLFW(const Config& config) : Adapter(config) {
    JST_DEBUG("[Metal] Creating GLFW viewport.");
}

Implementation::~GLFW() {
    JST_DEBUG("[Metal] Destroying GLFW viewport.");
}

Result Implementation::create() {
    // Register signal handler.

    keepRunningFlag = true;
    std::signal(SIGINT, [](int){
        if (!keepRunningFlag) {
            std::exit(0);
        }
        keepRunningFlag = false;
    });

    // Initialize and configure GLFW.

    if (!glfwInit()) {
        JST_ERROR("[Metal] Failed to initialize GLFW.");
        return Result::ERROR;
    }

    glfwSetErrorCallback(&PrintGLFWError);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DOUBLEBUFFER, config.vsync);

    auto [width, height] = config.size;
    window = glfwCreateWindow(width, height, 
        config.title.c_str(), nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        JST_ERROR("[Metal] Failed to create window with GLFW.");
        return Result::ERROR;
    }

    auto* device = Backend::State<Device::Metal>()->getDevice();

    swapchain = CA::MetalLayer::layer()->retain();
    swapchain->setDevice(device);
    swapchain->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    auto* native = static_cast<NS::Window*>(glfwGetCocoaWindow(window));
    native->contentView()->setLayer(swapchain);
    native->contentView()->setWantsLayer(true);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    glfwDestroyWindow(window);
    glfwTerminate();

    swapchain->release(); 

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    ImGui_ImplGlfw_InitForOther(window, true);

    return Result::SUCCESS;
}

F32 Implementation::scale(const F32& scale) const {
    // Scaling is handled gracefully by macOS/iOS.
    return scale;
}

Result Implementation::destroyImgui() {
    ImGui_ImplGlfw_Shutdown();

    return Result::SUCCESS;
}

void* Implementation::nextDrawable() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    swapchain->setDrawableSize({
        static_cast<CGFloat>(width), 
        static_cast<CGFloat>(height)
    });

    ImGui_ImplGlfw_NewFrame();

    return static_cast<void*>(swapchain->nextDrawable());
}

Result Implementation::waitEvents() {
    glfwWaitEventsTimeout(0.150);
    return Result::SUCCESS;
}

Result Implementation::pollEvents() {
    glfwPollEvents();
    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return !glfwWindowShouldClose(window) && keepRunningFlag;
}

}  // namespace Jetstream::Viewport 

