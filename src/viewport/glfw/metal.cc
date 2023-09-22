#include "jetstream/viewport/platforms/glfw/metal.hh"

namespace Jetstream::Viewport {

using Implementation = GLFW<Device::Metal>;

Implementation::GLFW(const Config& config) : Adapter(config) {
    JST_DEBUG("[Metal] Creating GLFW viewport.");
}

Implementation::~GLFW() {
    JST_DEBUG("[Metal] Destroying GLFW viewport.");
}

Result Implementation::create() {
    if (!glfwInit()) {
        JST_ERROR("[Metal] Failed to initialize GLFW.");
        return Result::ERROR;
    }

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

Result Implementation::pollEvents() {
    glfwWaitEvents();

    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return !glfwWindowShouldClose(window);
}

}  // namespace Jetstream::Viewport 

