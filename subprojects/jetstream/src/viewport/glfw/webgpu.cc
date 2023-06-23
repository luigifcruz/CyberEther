#include "jetstream/viewport/platforms/glfw/webgpu.hh"

static void PrintGLFWError(int error, const char* description) {
    JST_FATAL("[WebGPU] GLFW error: {}", description);
}

namespace Jetstream::Viewport {
    
using Implementation = GLFW<Device::WebGPU>;

Implementation::GLFW(const Config& config) : Adapter(config) {
    JST_DEBUG("[WebGPU] Creating GLFW viewport.");

    glfwSetErrorCallback(&PrintGLFWError);

    if (!glfwInit()) {
        JST_FATAL("[WebGPU] Failed to initialize GLFW.");
        JST_CHECK_THROW(Result::ERROR);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // glfwWindowHint(GLFW_DOUBLEBUFFER, config.vsync);

    auto [width, height] = config.size;
    window = glfwCreateWindow(width, height, 
        config.title.c_str(), nullptr, nullptr);
    swapchainSize = config.size;

    if (!window) {
        glfwTerminate();
        JST_FATAL("[WebGPU] Failed to create window with GLFW.");
        JST_CHECK_THROW(Result::ERROR);
    }

    wgpu::SurfaceDescriptorFromCanvasHTMLSelector html_surface_desc = {};
    html_surface_desc.selector = "#canvas";

    wgpu::SurfaceDescriptor surface_desc = {};
    surface_desc.nextInChain = &html_surface_desc;

    wgpu::Instance instance = {};
    surface = instance.CreateSurface(&surface_desc);

    glfwShowWindow(window);
};

Implementation::~GLFW() {
    JST_DEBUG("[WebGPU] Destroying GLFW viewport.");

    glfwDestroyWindow(window);
    glfwTerminate();
}

Result Implementation::create() {
    auto& device = Backend::State<Device::WebGPU>()->getDevice();

    wgpu::SwapChainDescriptor swapchainDesc = {};
    swapchainDesc.usage = wgpu::TextureUsage::RenderAttachment;
    swapchainDesc.format = wgpu::TextureFormat::BGRA8Unorm;
    swapchainDesc.width = swapchainSize.width;
    swapchainDesc.height = swapchainSize.height;
    swapchainDesc.presentMode = wgpu::PresentMode::Fifo;
    swapchain = device.CreateSwapChain(surface, &swapchainDesc);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    swapchain.Release();
    
    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    ImGui_ImplGlfw_InitForOther(window, true);

    return Result::SUCCESS;
}
Result Implementation::destroyImgui() {
    ImGui_ImplGlfw_Shutdown();

    return Result::SUCCESS;
}

Result Implementation::nextDrawable() {
    ImGui_ImplGlfw_NewFrame();

    return Result::SUCCESS;
}

Result Implementation::commitDrawable(wgpu::TextureView& framebufferTexture) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    if (width != swapchainSize.width && height != swapchainSize.height) {
        swapchainSize = {static_cast<U64>(width), static_cast<U64>(height)};
        return Result::RECREATE;
    }

    framebufferTexture = swapchain.GetCurrentTextureView();
    
    return Result::SUCCESS;
}

Result Implementation::pollEvents() {
    glfwWaitEvents();

    return Result::SUCCESS;
}

bool Implementation::keepRunning() {
    return !glfwWindowShouldClose(window);
}

}  // namespace Jetstream::Viewport
