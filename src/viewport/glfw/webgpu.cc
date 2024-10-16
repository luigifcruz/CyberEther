#include "jetstream/viewport/platforms/glfw/webgpu.hh"

#include <GLFW/glfw3.h>

#include "tools/imgui_impl_glfw.h"

EM_JS(int, getWindowWidth, (), {
    return window.innerWidth;
});

EM_JS(int, getWindowHeight, (), {
    return window.innerHeight;
});

EM_JS(int, getPixelRatio, (), {
    return window.devicePixelRatio;
});

static void PrintGLFWError(int, const char* description) {
    JST_FATAL("[WebGPU] GLFW error: {}", description);
}

namespace Jetstream::Viewport {
    
using Implementation = GLFW<Device::WebGPU>;

Implementation::GLFW(const Config& config) : Adapter(config) {
    JST_DEBUG("[WebGPU] Creating GLFW viewport.");
};

Implementation::~GLFW() {
    JST_DEBUG("[WebGPU] Destroying GLFW viewport.");
}

Result Implementation::create() {
    glfwSetErrorCallback(&PrintGLFWError);

    if (!glfwInit()) {
        JST_ERROR("[WebGPU] Failed to initialize GLFW.");
        return Result::ERROR;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);

    swapchainSize = {
        static_cast<U64>(getWindowWidth()),
        static_cast<U64>(getWindowHeight())
    };

    window = glfwCreateWindow(swapchainSize.x, swapchainSize.y, config.title.c_str(), nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        JST_ERROR("[WebGPU] Failed to create window with GLFW.");
        return Result::ERROR;
    }
    glfwMakeContextCurrent(window);

    wgpu::SurfaceDescriptorFromCanvasHTMLSelector html_surface_desc{};
    html_surface_desc.selector = "#canvas";

    wgpu::SurfaceDescriptor surface_desc{};
    surface_desc.nextInChain = &html_surface_desc;

    instance = wgpu::CreateInstance(nullptr);

    surface = instance.CreateSurface(&surface_desc);

    JST_CHECK(createSwapchain());

    glfwShowWindow(window);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_CHECK(destroySwapchain());

    glfwDestroyWindow(window);
    glfwTerminate();

    return Result::SUCCESS;
}

Result Implementation::createSwapchain() {
    glfwSetWindowSize(window, swapchainSize.x, swapchainSize.y);

    wgpu::SwapChainDescriptor swapchainDesc{};
    swapchainDesc.usage = wgpu::TextureUsage::RenderAttachment;
    swapchainDesc.format = wgpu::TextureFormat::BGRA8Unorm;
    swapchainDesc.width = swapchainSize.x * getPixelRatio();
    swapchainDesc.height = swapchainSize.y * getPixelRatio();
    swapchainDesc.presentMode = wgpu::PresentMode::Fifo;

    auto& device = Backend::State<Device::WebGPU>()->getDevice();
    swapchain = device.CreateSwapChain(surface, &swapchainDesc);

    return Result::SUCCESS;
}

Result Implementation::destroySwapchain() {
    swapchain = nullptr;
    
    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    ImGui_ImplGlfw_InitForOther(window, true);

    return Result::SUCCESS;
}

F32 Implementation::scale(const F32& scale) const {
    // No scaling needed. ImGui was modified to handle HiDPI. 
    return scale;
}

Result Implementation::destroyImgui() {
    ImGui_ImplGlfw_Shutdown();

    return Result::SUCCESS;
}

Result Implementation::nextDrawable() {
    U64 width = getWindowWidth();
    U64 height = getWindowHeight();

    if (width != swapchainSize.x or height != swapchainSize.y) {
       swapchainSize = {static_cast<U64>(width), static_cast<U64>(height)};
       return Result::RECREATE;
    }

    ImGui_ImplGlfw_NewFrame();

    return Result::SUCCESS;
}

Result Implementation::commitDrawable(wgpu::TextureView& framebufferTexture) {
    framebufferTexture = swapchain.GetCurrentTextureView();
    
    return Result::SUCCESS;
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
    return !glfwWindowShouldClose(window);
}

}  // namespace Jetstream::Viewport
