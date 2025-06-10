#include "jetstream/render/devices/webgpu/window.hh"
#include "jetstream/render/devices/webgpu/surface.hh"
#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/texture.hh"

#include "tools/imgui_impl_wgpu.h"

namespace Jetstream::Render {

using Implementation = WindowImp<Device::WebGPU>;

Implementation::WindowImp(const Config& config,
                          std::shared_ptr<Viewport::Adapter<Device::WebGPU>>& viewport)
         : Window(config), viewport(viewport) {
}

template<typename T>
Result Implementation::bindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container) {
    // Cast generic resource.
    auto _resource = std::dynamic_pointer_cast<T>(resource);

    // Create the resource.
    JST_CHECK(_resource->create());

    // Add resource to container.
    container.push_back(_resource);

    return Result::SUCCESS;
}

template<typename T>
Result Implementation::unbindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container) {
    // Cast generic resource.
    auto _resource = std::dynamic_pointer_cast<T>(resource);

    // Destroy the resource.
    JST_CHECK(_resource->destroy());

    // Remove resource from container.
    container.erase(std::remove(container.begin(), container.end(), _resource), container.end());

    return Result::SUCCESS;
}

Result Implementation::bindBuffer(const std::shared_ptr<Buffer>& buffer) {
    return bindResource<BufferImp<Device::WebGPU>>(buffer, buffers);
}

Result Implementation::unbindBuffer(const std::shared_ptr<Buffer>& buffer) {
    return unbindResource<BufferImp<Device::WebGPU>>(buffer, buffers);
}

Result Implementation::bindTexture(const std::shared_ptr<Texture>& texture) {
    return bindResource<TextureImp<Device::WebGPU>>(texture, textures);
}

Result Implementation::unbindTexture(const std::shared_ptr<Texture>& texture) {
    return unbindResource<TextureImp<Device::WebGPU>>(texture, textures);
}

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    return bindResource<SurfaceImp<Device::WebGPU>>(surface, surfaces);
}

Result Implementation::unbindSurface(const std::shared_ptr<Surface>& surface) {
    return unbindResource<SurfaceImp<Device::WebGPU>>(surface, surfaces);
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

    JST_CHECK(destroyImgui());

    if (!buffers.empty() || !textures.empty() || !surfaces.empty()) {
        JST_WARN("[WebGPU] Resources are still bounded to this window "
                 "(buffers={}, textures={}, surfaces={}).", 
                 buffers.size(), textures.size(), surfaces.size());
    }

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
