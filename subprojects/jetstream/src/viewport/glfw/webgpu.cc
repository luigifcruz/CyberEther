#include "jetstream/viewport/platforms/glfw/webgpu.hh"

EM_JS(int, getWindowWidth, (), {
    return window.innerWidth;
});

EM_JS(int, getWindowHeight, (), {
    return window.innerHeight;
});

EM_JS(int, getPixelRatio, (), {
    return window.devicePixelRatio;
});

static void PrintGLFWError(int error, const char* description) {
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
        JST_FATAL("[WebGPU] Failed to initialize GLFW.");
        JST_CHECK_THROW(Result::ERROR);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    swapchainSize = {
        static_cast<U64>(getWindowWidth()),
        static_cast<U64>(getWindowHeight())
    };

    window = glfwCreateWindow(swapchainSize.width, swapchainSize.height, config.title.c_str(), nullptr, nullptr);

    if (!window) {
        glfwTerminate();
        JST_FATAL("[WebGPU] Failed to create window with GLFW.");
        JST_CHECK_THROW(Result::ERROR);
    }
    glfwMakeContextCurrent(window);

    wgpu::SurfaceDescriptorFromCanvasHTMLSelector html_surface_desc{};
    html_surface_desc.selector = "#canvas";

    wgpu::SurfaceDescriptor surface_desc{};
    surface_desc.nextInChain = &html_surface_desc;

    wgpu::Instance instance{};
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
    glfwSetWindowSize(window, swapchainSize.width, swapchainSize.height);

    wgpu::SwapChainDescriptor swapchainDesc{};
    swapchainDesc.usage = wgpu::TextureUsage::RenderAttachment;
    swapchainDesc.format = wgpu::TextureFormat::BGRA8Unorm;
    swapchainDesc.width = swapchainSize.width * getPixelRatio();
    swapchainDesc.height = swapchainSize.height * getPixelRatio();
    swapchainDesc.presentMode = wgpu::PresentMode::Fifo;

    auto& device = Backend::State<Device::WebGPU>()->getDevice();
    swapchain = device.CreateSwapChain(surface, &swapchainDesc);

    return Result::SUCCESS;
}

Result Implementation::destroySwapchain() {
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
    int width = getWindowWidth();
    int height = getWindowHeight();

    if (width != swapchainSize.width or height != swapchainSize.height) {
       swapchainSize = {static_cast<U64>(width), static_cast<U64>(height)};
       return Result::RECREATE;
    }

    framebufferTexture = swapchain.GetCurrentTextureView();
    
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
