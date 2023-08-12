#include "jetstream/render/webgpu/window.hh"
#include "jetstream/render/webgpu/surface.hh"

namespace Jetstream::Render {

using Implementation = WindowImp<Device::WebGPU>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Adapter<Device::WebGPU>>& viewport)
         : Window(config), viewport(viewport) {
}

Result Implementation::bind(const std::shared_ptr<Surface>& surface) {
    JST_DEBUG("[WebGPU] Binding surface to window.");

    surfaces.push_back(
        std::dynamic_pointer_cast<SurfaceImp<Device::WebGPU>>(surface)
    );

    return Result::SUCCESS;
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating window.");

    auto& device = Backend::State<Device::WebGPU>()->getDevice();

    queue = device.GetQueue();    

    for (auto& surface : surfaces) {
        JST_CHECK(surface->create());
    }

    if (config.imgui) {
        JST_CHECK(createImgui());
    }

    statsData.droppedFrames = 0;
    
    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying window.");

    for (auto& surface : surfaces) {
        JST_CHECK(surface->destroy());
    }

    if (config.imgui) {
        JST_CHECK(destroyImgui());
    } 

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    JST_DEBUG("[WebGPU] Creating ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImNodes::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    auto& device = Backend::State<Device::WebGPU>()->getDevice();
    ImGui_ImplWGPU_Init(device.Get(), 3, WGPUTextureFormat_BGRA8Unorm, WGPUTextureFormat_Undefined);
    
    JST_CHECK(viewport->createImgui());

    const auto& scale = viewport->calculateScale(config.scale);
    ApplyImGuiTheme(scale);
    ApplyImNodesTheme(scale);

    ImGui_ImplWGPU_CreateDeviceObjects();
    
    return Result::SUCCESS;
}

Result Implementation::destroyImgui() {
    JST_DEBUG("[WebGPU] Destroying ImGui.");

    ImGui_ImplWGPU_InvalidateDeviceObjects();
    ImGui_ImplWGPU_Shutdown();
    JST_CHECK(viewport->destroyImgui());
    ImNodes::DestroyContext();
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

Result Implementation::recreate() {
    JST_CHECK(destroy());
    JST_CHECK(viewport->destroySwapchain());
    JST_CHECK(viewport->createSwapchain());
    JST_CHECK(create());

    return Result::SUCCESS;
}

Result Implementation::beginImgui() {
    ImGui_ImplWGPU_NewFrame();

    ApplyImGuiScale();
    ApplyImNodesScale();

    ImGui::NewFrame();

    return Result::SUCCESS;
}

Result Implementation::endImgui() {
    ImGui::Render();

    auto renderPassEncoder = encoder.BeginRenderPass(&renderPassDesc).Release();
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPassEncoder);
    wgpuRenderPassEncoderEnd(renderPassEncoder);

    return Result::SUCCESS;
}

Result Implementation::begin() {
    JST_CHECK(viewport->nextDrawable());
        
    if (config.imgui) {
        JST_CHECK(beginImgui());
    }

    return Result::SUCCESS;
}

Result Implementation::end() {
    wgpu::TextureView framebufferTexture;
    const Result& result = viewport->commitDrawable(framebufferTexture);

    if (result == Result::SKIP) {
        statsData.droppedFrames += 1;
        return Result::SKIP;
    } else if (result == Result::RECREATE) {
        JST_CHECK(recreate());
        return Result::SKIP;
    } else if (result != Result::SUCCESS) {
        return result;
    }

    auto& device = Backend::State<Device::WebGPU>()->getDevice();

    colorAttachments = {};
    colorAttachments.loadOp = wgpu::LoadOp::Clear;
    colorAttachments.storeOp = wgpu::StoreOp::Store;
    colorAttachments.clearValue.r = 0.0f;
    colorAttachments.clearValue.g = 0.0f;
    colorAttachments.clearValue.b = 0.0f;
    colorAttachments.clearValue.a = 1.0f;
    colorAttachments.view = framebufferTexture;

    renderPassDesc = {};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachments;
    renderPassDesc.depthStencilAttachment = nullptr;

    wgpu::CommandEncoderDescriptor encDesc{};
    encoder = device.CreateCommandEncoder(&encDesc);

    if (config.imgui) {
        JST_CHECK(endImgui());
    }

    for (auto &surface : surfaces) {
        JST_CHECK(surface->draw(encoder));
    }

    wgpu::CommandBufferDescriptor cmdBufferDesc{};
    wgpu::CommandBuffer cmdBuffer = encoder.Finish(&cmdBufferDesc);
    device.GetQueue().Submit(1, &cmdBuffer);

    return Result::SUCCESS;
}

void Implementation::drawDebugMessage() const {
    auto& backend = Backend::State<Device::WebGPU>();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::TextFormatted("FPS: {:.1f} Hz", io.Framerate);
    ImGui::TextFormatted("Device Name: {}", backend->getDeviceName());
    ImGui::TextFormatted("Low Power Mode: {}", backend->getLowPowerStatus() ? "YES" : "NO");
    ImGui::TextFormatted("Has Unified Memory: {}", backend->hasUnifiedMemory() ? "YES" : "NO");
    ImGui::TextFormatted("Physical Memory: {:.0f} GB", (float)backend->getPhysicalMemory() / (1024*1024*1024));
    ImGui::TextFormatted("Thermal State: {}/3", backend->getThermalState());
    ImGui::TextFormatted("Processor Count: {}", backend->getTotalProcessorCount());
}

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
