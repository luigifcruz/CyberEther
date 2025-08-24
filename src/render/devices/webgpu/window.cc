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

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<Device::WebGPU>>(surface);
    surfaces.push_back(_resource);
    return Result::SUCCESS;
}

Result Implementation::unbindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<Device::WebGPU>>(surface);
    surfaces.erase(std::remove(surfaces.begin(), surfaces.end(), _resource), surfaces.end());
    return Result::SUCCESS;
}

Result Implementation::underlyingCreate() {
    JST_DEBUG("[WebGPU] Creating window.");


    auto device = Backend::State<Device::WebGPU>()->getDevice();


    queue = wgpuDeviceGetQueue(device);

    JST_CHECK(createImgui());

    // Reseting internal variables.

    statsData.droppedFrames = 0;
    statsData.recreatedFrames = 0;

    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
    JST_DEBUG("[WebGPU] Destroying window.");

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


    auto device = Backend::State<Device::WebGPU>()->getDevice();

    ImGui_ImplWGPU_InitInfo info;
    info.Device = device;
    info.NumFramesInFlight = 3;
    info.RenderTargetFormat = WGPUTextureFormat_BGRA8Unorm;
    info.DepthStencilFormat = WGPUTextureFormat_Undefined;
    ImGui_ImplWGPU_Init(&info);

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

    // Begin the render pass using the C API.
    WGPURenderPassEncoder renderPassEncoder = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPassEncoder);
    wgpuRenderPassEncoderEnd(renderPassEncoder);

    return Result::SUCCESS;
}

Result Implementation::underlyingBegin() {
    const auto& result = viewport->nextDrawable();

    if (result == Result::SKIP) {
        statsData.droppedFrames += 1;
        return Result::SKIP;
    }

    if (result == Result::RECREATE) {
        statsData.recreatedFrames += 1;
        JST_CHECK(recreate());
        return Result::SKIP;
    }

    if (result != Result::SUCCESS) {
        JST_FATAL("[VULKAN] Failed to acquire next viewport drawable.");
        return Result::ERROR;
    }


    WGPUTextureView framebufferTexture = {};
    JST_CHECK(viewport->commitDrawable(&framebufferTexture));

    colorAttachments = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    colorAttachments.loadOp = WGPULoadOp_Clear;
    colorAttachments.storeOp = WGPUStoreOp_Store;
    colorAttachments.clearValue.r = 0.0f;
    colorAttachments.clearValue.g = 0.0f;
    colorAttachments.clearValue.b = 0.0f;
    colorAttachments.clearValue.a = 1.0f;
    colorAttachments.view = framebufferTexture;


    renderPassDesc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachments;
    renderPassDesc.depthStencilAttachment = nullptr;


    auto device = Backend::State<Device::WebGPU>()->getDevice();
    WGPUCommandEncoderDescriptor encDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

    // Let each bound surface draw into the command encoder.
    for (auto &surface : surfaces) {
        JST_CHECK(surface->draw(encoder));
    }

    JST_CHECK(beginImgui());

    return Result::SUCCESS;
}

Result Implementation::underlyingEnd() {
    JST_CHECK(endImgui());


    WGPUCommandBufferDescriptor cmdBufferDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
    wgpuQueueSubmit(queue, 1, &cmdBuffer);

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
