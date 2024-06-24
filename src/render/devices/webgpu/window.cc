#include "jetstream/render/devices/webgpu/window.hh"
#include "jetstream/render/devices/webgpu/surface.hh"

#include "tools/imgui_impl_wgpu.h"

namespace Jetstream::Render {

using Implementation = WindowImp<Device::WebGPU>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Adapter<Device::WebGPU>>& viewport)
         : Window(config), viewport(viewport) {
}

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    JST_DEBUG("[WebGPU] Binding surface to window.");

    // Cast generic Surface.
    auto _surface = std::dynamic_pointer_cast<SurfaceImp<Device::WebGPU>>(surface);

    // Create the Surface.
    JST_CHECK(_surface->create());

    // Add Surface to window.
    surfaces.push_back(_surface);

    return Result::SUCCESS;
}

Result Implementation::unbindSurface(const std::shared_ptr<Surface>& surface) {
    JST_DEBUG("[WebGPU] Unbinding surface from window.");

    // Cast generic Surface.
    auto _surface = std::dynamic_pointer_cast<SurfaceImp<Device::WebGPU>>(surface);

    // Destroy the Surface.
    JST_CHECK(_surface->destroy());

    // Remove Surface from window.
    surfaces.erase(std::remove(surfaces.begin(), surfaces.end(), _surface), surfaces.end());

    return Result::SUCCESS;
}

Result Implementation::underlyingCreate() {
    JST_DEBUG("[WebGPU] Creating window.");

    auto& device = Backend::State<Device::WebGPU>()->getDevice();

    queue = device.GetQueue();    

    JST_CHECK(createImgui());

    statsData.droppedFrames = 0;
    
    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
    JST_DEBUG("[WebGPU] Destroying window.");

    for (auto& surface : surfaces) {
        JST_CHECK(surface->destroy());
    }

    JST_CHECK(destroyImgui());

    return Result::SUCCESS;
}

Result Implementation::createImgui() {
    JST_DEBUG("[WebGPU] Creating ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();

    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    JST_CHECK(viewport->createImgui());

    this->scaleStyle(*viewport);

    auto& device = Backend::State<Device::WebGPU>()->getDevice();
    ImGui_ImplWGPU_Init(device.Get(), 3, WGPUTextureFormat_BGRA8Unorm, WGPUTextureFormat_Undefined);
    
    return Result::SUCCESS;
}

Result Implementation::destroyImgui() {
    JST_DEBUG("[WebGPU] Destroying ImGui.");

    ImGui_ImplWGPU_InvalidateDeviceObjects();
    ImGui_ImplWGPU_Shutdown();
    JST_CHECK(viewport->destroyImgui());
    ImGui::DestroyContext();

    return Result::SUCCESS;
}

Result Implementation::recreate() {
    JST_CHECK(viewport->destroySwapchain());
    JST_CHECK(viewport->createSwapchain());

    return Result::SUCCESS;
}

Result Implementation::beginImgui() {
    ImGui_ImplWGPU_NewFrame();

    this->scaleStyle(*viewport);

    ImGui::NewFrame();

    return Result::SUCCESS;
}

Result Implementation::endImgui() {
    ImGui::Render();

    auto renderPassEncoder = encoder.BeginRenderPass(&renderPassDesc).MoveToCHandle();
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPassEncoder);
    wgpuRenderPassEncoderEnd(renderPassEncoder);

    return Result::SUCCESS;
}

Result Implementation::underlyingBegin() {
    const auto& result = viewport->nextDrawable();

    if (result == Result::SKIP) {
        statsData.droppedFrames += 1;
        return Result::SKIP;
    } else if (result == Result::RECREATE) {
        JST_CHECK(recreate());
        return Result::SKIP;
    } else if (result != Result::SUCCESS) {
        return result;
    }

    wgpu::TextureView framebufferTexture;
    JST_CHECK(viewport->commitDrawable(framebufferTexture));

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

    auto& device = Backend::State<Device::WebGPU>()->getDevice();
    wgpu::CommandEncoderDescriptor encDesc{};
    encoder = device.CreateCommandEncoder(&encDesc);

    for (auto &surface : surfaces) {
        JST_CHECK(surface->draw(encoder));
    }

    JST_CHECK(beginImgui());

    return Result::SUCCESS;
}

Result Implementation::underlyingEnd() {
    JST_CHECK(endImgui());

    auto& device = Backend::State<Device::WebGPU>()->getDevice();
    wgpu::CommandBufferDescriptor cmdBufferDesc{};
    wgpu::CommandBuffer cmdBuffer = encoder.Finish(&cmdBufferDesc);
    device.GetQueue().Submit(1, &cmdBuffer);

    return Result::SUCCESS;
}

Result Implementation::underlyingSynchronize() {
    return Result::SUCCESS;
}

void Implementation::drawDebugMessage() const {
    auto& backend = Backend::State<Device::WebGPU>();

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Device Name:");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-1);
    ImGui::TextFormatted("{}", backend->getDeviceName());
}

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
