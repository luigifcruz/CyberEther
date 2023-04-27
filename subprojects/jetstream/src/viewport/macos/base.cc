#include "jetstream/viewport/macos.hh"

namespace Jetstream::Viewport {

MacOS::MacOS(const Config& config) : Generic(config) {
    JST_DEBUG("Creating macOS viewport.");
}

MacOS::~MacOS() {
    JST_DEBUG("Destroying macOS viewport.");
}

Result MacOS::create() {
    if (!glfwInit()) {
        return Result::ERROR;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DOUBLEBUFFER, config.vsync);

    auto [width, height] = config.size;
    window = glfwCreateWindow(width, height, 
        config.title.c_str(), nullptr, nullptr);

    if (!window) {
        glfwTerminate();
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

Result MacOS::destroy() {
    glfwDestroyWindow(window);
    glfwTerminate();

    swapchain->release(); 

    return Result::SUCCESS;
}

Result MacOS::createImgui() {
    ImGui_ImplGlfw_InitForOther(window, true);

    return Result::SUCCESS;
}

Result MacOS::destroyImgui() {
    ImGui_ImplGlfw_Shutdown();

    return Result::SUCCESS;
}

void* MacOS::nextDrawable() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    swapchain->setDrawableSize({
        static_cast<CGFloat>(width), 
        static_cast<CGFloat>(height)
    });

    ImGui_ImplGlfw_NewFrame();

    return static_cast<void*>(swapchain->nextDrawable());
}

Result MacOS::pollEvents() {
    glfwWaitEvents();

    return Result::SUCCESS;
}

bool MacOS::keepRunning() {
    return !glfwWindowShouldClose(window);
}

}  // namespace Jetstream::Viewport 

