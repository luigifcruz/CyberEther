#include "jetstream/render/devices/webgpu/window.hh"
#include "jetstream/render/devices/webgpu/surface.hh"

#include "tools/imgui_impl_wgpu.h"

namespace Jetstream::Render {

using Implementation = WindowImp<DeviceType::WebGPU>;

Implementation::WindowImp(const Config& config,
                          const std::shared_ptr<Viewport::Adapter<DeviceType::WebGPU>>& viewport)
         : Window(config), viewport(viewport) {
}

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<DeviceType::WebGPU>>(surface);
    surfaces.push_back(_resource);
    return Result::SUCCESS;
}

Result Implementation::unbindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<DeviceType::WebGPU>>(surface);
    surfaces.erase(std::remove(surfaces.begin(), surfaces.end(), _resource), surfaces.end());
    return Result::SUCCESS;
}

Result Implementation::underlyingCreate() {
    JST_DEBUG("[WebGPU] Creating window.");


    auto device = Backend::State<DeviceType::WebGPU>()->getDevice();


    queue = wgpuDeviceGetQueue(device);

    JST_CHECK(createImgui());

    // Reseting internal variables.

    statsData.droppedFrames = 0;
    statsData.recreatedFrames = 0;

    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
    JST_DEBUG("[WebGPU] Destroying window.");

    Result result = underlyingCancel();

    if (imguiCreated) {
        const Result imguiResult = destroyImgui();
        if ((result == Result::SUCCESS || result == Result::RELOAD) &&
            imguiResult != Result::SUCCESS && imguiResult != Result::RELOAD) {
            result = imguiResult;
        }
    }

    transferEncoder.destroy();

    if (queue) {
        wgpuQueueRelease(queue);
        queue = nullptr;
    }

    return result;
}

Result Implementation::createImgui() {
    JST_DEBUG("[WebGPU] Creating ImGui.");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    io = &ImGui::GetIO();
    style = &ImGui::GetStyle();
    io->IniFilename = nullptr;

    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    Result viewportResult;
    try {
        viewportResult = viewport->createImgui();
    } catch (...) {
        ImGui::DestroyContext();
        io = nullptr;
        style = nullptr;
        throw;
    }
    if (viewportResult != Result::SUCCESS && viewportResult != Result::RELOAD) {
        ImGui::DestroyContext();
        io = nullptr;
        style = nullptr;
        return viewportResult;
    }

    try {
        this->updateScalingFactor(*viewport);

        auto device = Backend::State<DeviceType::WebGPU>()->getDevice();

        ImGui_ImplWGPU_InitInfo info{};
        info.Device = device;
        info.NumFramesInFlight = 3;
        info.RenderTargetFormat = WGPUTextureFormat_BGRA8Unorm;
        info.DepthStencilFormat = WGPUTextureFormat_Undefined;
        ImGui_ImplWGPU_Init(&info);
    } catch (...) {
        viewport->destroyImgui();
        ImGui::DestroyContext();
        io = nullptr;
        style = nullptr;
        throw;
    }
    imguiCreated = true;

    return Result::SUCCESS;
}

Result Implementation::destroyImgui() {
    JST_DEBUG("[WebGPU] Destroying ImGui.");

    ImGui_ImplWGPU_InvalidateDeviceObjects();
    ImGui_ImplWGPU_Shutdown();
    const Result result = viewport->destroyImgui();
    ImGui::DestroyContext();
    io = nullptr;
    style = nullptr;
    imguiCreated = false;

    return result;
}

Result Implementation::recreate() {
    JST_CHECK(viewport->destroySwapchain());
    JST_CHECK(viewport->createSwapchain());

    return Result::SUCCESS;
}

Result Implementation::beginImgui() {
    ImGui_ImplWGPU_NewFrame();

    this->updateScalingFactor(*viewport);

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

    framebufferTexture = {};
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

    auto device = Backend::State<DeviceType::WebGPU>()->getDevice();
    WGPUCommandEncoderDescriptor encDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
    if (!encoder) {
        wgpuTextureViewRelease(framebufferTexture);
        framebufferTexture = nullptr;
        return Result::ERROR;
    }

    for (auto& surface : surfaces) {
        const Result prepareResult = surface->prepare();
        if (prepareResult != Result::SUCCESS && prepareResult != Result::RELOAD) {
            wgpuCommandEncoderRelease(encoder);
            encoder = nullptr;
            wgpuTextureViewRelease(framebufferTexture);
            framebufferTexture = nullptr;
            return prepareResult;
        }
    }

    const Result beginResult = beginImgui();
    if (beginResult != Result::SUCCESS && beginResult != Result::RELOAD) {
        wgpuCommandEncoderRelease(encoder);
        encoder = nullptr;
        wgpuTextureViewRelease(framebufferTexture);
        framebufferTexture = nullptr;
        return beginResult;
    }

    return Result::SUCCESS;
}

Result Implementation::underlyingEnd() {
    Transfer::Batch transfers;

    const auto abortFrame = [&](const Result& result) {
        const Result recovery = underlyingCancel();
        return recovery == Result::SUCCESS ? result : recovery;
    };

    Result result = collectTransfers(transfers);
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        return abortFrame(result);
    }
    if (!transfers.empty()) {
        result = transferEncoder.encode(transfers, queue, encoder);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    for (auto& surface : surfaces) {
        result = surface->draw(encoder);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    result = endImgui();
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        return abortFrame(result);
    }

    WGPUCommandBufferDescriptor cmdBufferDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
    if (!cmdBuffer) {
        return abortFrame(Result::ERROR);
    }
    wgpuQueueSubmit(queue, 1, &cmdBuffer);
    for (auto& surface : surfaces) {
        surface->commitDraw();
    }
    transfers.commit();

    wgpuCommandBufferRelease(cmdBuffer);
    wgpuCommandEncoderRelease(encoder);
    encoder = nullptr;

    if (framebufferTexture) {
        wgpuTextureViewRelease(framebufferTexture);
        framebufferTexture = nullptr;
    }

    return Result::SUCCESS;
}

Result Implementation::underlyingCancel() {
    if (encoder) {
        wgpuCommandEncoderRelease(encoder);
        encoder = nullptr;
    }
    if (framebufferTexture) {
        wgpuTextureViewRelease(framebufferTexture);
        framebufferTexture = nullptr;
    }
    return Result::SUCCESS;
}

Result Implementation::underlyingSynchronize() {
    return Result::SUCCESS;
}

std::string Implementation::info() const {
    // WebGPU doesn't expose any useful device information.
    return "WebGPU Device";
}

const Window::Stats& Implementation::stats() const {
    return statsData;
}

}  // namespace Jetstream::Render
