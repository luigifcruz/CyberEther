#include "jetstream/viewport/platforms/glfw/webgpu.hh"

#include <GLFW/glfw3.h>
#include <GLFW/emscripten_glfw3.h>
#include <emscripten/html5.h>

#include "tools/imgui_impl_glfw.h"

static void PrintGLFWError(int, const char* description) {
    JST_FATAL("[WebGPU] GLFW error: {}", description);
}

namespace Jetstream::Viewport {

using Implementation = GLFW<DeviceType::WebGPU>;

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
    glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_TRUE);

    double width = 1280.0;
    double height = 720.0;
    if (emscripten_get_element_css_size("#canvas", &width, &height) != EMSCRIPTEN_RESULT_SUCCESS) {
        JST_WARN("[WebGPU] Failed to query canvas size; using default dimensions.");
    }

    emscripten::glfw3::SetNextWindowCanvasSelector("#canvas");
    window = glfwCreateWindow(static_cast<int>(width),
                              static_cast<int>(height),
                              config.title.c_str(),
                              nullptr,
                              nullptr);

    if (!window) {
        glfwTerminate();
        JST_ERROR("[WebGPU] Failed to create window with GLFW.");
        return Result::ERROR;
    }

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    swapchainSize = {
        static_cast<U64>(framebufferWidth),
        static_cast<U64>(framebufferHeight),
    };

    WGPUSurfaceDescriptor surfaceDesc = WGPU_SURFACE_DESCRIPTOR_INIT;

    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector surfaceSource = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    surfaceSource.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
    surfaceSource.selector = {"#canvas", WGPU_STRLEN};

    surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&surfaceSource);

    instance = Backend::State<DeviceType::WebGPU>()->getInstance();

    surface = wgpuInstanceCreateSurface(instance, &surfaceDesc);
    if (!surface) {
        glfwDestroyWindow(window);
        glfwTerminate();
        JST_ERROR("[WebGPU] Failed to create canvas surface.");
        return Result::ERROR;
    }

    JST_CHECK(createSwapchain());

    glfwShowWindow(window);

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_CHECK(destroySwapchain());

    if (surface) {
        wgpuSurfaceRelease(surface);
        surface = nullptr;
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return Result::SUCCESS;
}

Result Implementation::createSwapchain() {
    auto device = Backend::State<DeviceType::WebGPU>()->getDevice();

    WGPUSurfaceConfiguration conf = WGPU_SURFACE_CONFIGURATION_INIT;
    conf.device = device;
    conf.usage = WGPUTextureUsage_RenderAttachment;
    conf.format = WGPUTextureFormat_BGRA8Unorm;
    conf.width = swapchainSize.x;
    conf.height = swapchainSize.y;
    conf.presentMode = WGPUPresentMode_Fifo;
    wgpuSurfaceConfigure(surface, &conf);

    return Result::SUCCESS;
}

Result Implementation::destroySwapchain() {
    wgpuSurfaceUnconfigure(surface);

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    ImGui_ImplGlfw_InitForOther(window, true);
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");

    return Result::SUCCESS;
}

Extent2D<F32> Implementation::displaySize() const {
    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);
    return {static_cast<F32>(width), static_cast<F32>(height)};
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
    glfwPollEvents();

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    if (framebufferWidth <= 0 || framebufferHeight <= 0) {
        return Result::SKIP;
    }

    if (static_cast<U64>(framebufferWidth) != swapchainSize.x ||
        static_cast<U64>(framebufferHeight) != swapchainSize.y) {
       swapchainSize = {
           static_cast<U64>(framebufferWidth),
           static_cast<U64>(framebufferHeight),
       };
       return Result::RECREATE;
    }

    ImGui_ImplGlfw_NewFrame();

    return Result::SUCCESS;
}

Result Implementation::commitDrawable(WGPUTextureView* framebufferTexture) {
    WGPUSurfaceTexture st = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(surface, &st);

    if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        *framebufferTexture = nullptr;
        return Result::RECREATE;
    }

    *framebufferTexture = wgpuTextureCreateView(st.texture, nullptr);

    return (*framebufferTexture != nullptr) ? Result::SUCCESS : Result::RECREATE;
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
