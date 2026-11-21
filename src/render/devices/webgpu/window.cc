#include <algorithm>

#include "jetstream/render/devices/webgpu/window.hh"
#include "jetstream/render/devices/webgpu/surface.hh"

#include "presenter.hh"

namespace Jetstream::Render {

using Implementation = WindowImp<DeviceType::WebGPU>;

Implementation::WindowImp(const Config& config,
                          const std::shared_ptr<Viewport::Adapter<DeviceType::WebGPU>>& viewport)
         : Window(config), viewport(viewport) {
}

Implementation::~WindowImp() = default;

Result Implementation::bindSurface(const std::shared_ptr<Surface>& surface) {
    auto _resource = std::dynamic_pointer_cast<SurfaceImp<DeviceType::WebGPU>>(surface);
    if (_resource->getConfig().composite) {
        surfaces.push_back(_resource);
    } else {
        const auto firstComposite = std::find_if(surfaces.begin(), surfaces.end(), [](const auto& other) {
            return other->getConfig().composite;
        });
        surfaces.insert(firstComposite, _resource);
    }
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

    JST_CHECK(createPresenter());
    this->updateMetrics(*viewport);

    return Result::SUCCESS;
}

Result Implementation::underlyingDestroy() {
    JST_DEBUG("[WebGPU] Destroying window.");

    Result result = underlyingCancel();

    if (presenter) {
        presenter->destroy();
        presenter.reset();
    }

    transferEncoder.destroy();

    if (queue) {
        wgpuQueueRelease(queue);
        queue = nullptr;
    }

    return result;
}

Result Implementation::createPresenter() {
    presenter = std::make_unique<Presenter>();
    const Result result = presenter->create(WGPUTextureFormat_BGRA8Unorm);
    if (result != Result::SUCCESS) {
        presenter.reset();
    }
    return result;
}

Result Implementation::underlyingRecreate() {
    JST_CHECK(viewport->destroySwapchain());
    JST_CHECK(viewport->createSwapchain());

    return Result::SUCCESS;
}

Result Implementation::encodePresent() {
    const auto& background = frameClearColor();

    WGPURenderPassColorAttachment colorAttachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    colorAttachment.loadOp = WGPULoadOp_Clear;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.view = framebufferTexture;
    colorAttachment.clearValue = {background.r, background.g, background.b, background.a};

    WGPURenderPassDescriptor renderPassDesc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;
    renderPassDesc.depthStencilAttachment = nullptr;

    WGPURenderPassEncoder renderPassEncoder = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
    const Result result = presenter->encode(renderPassEncoder, queue, framePresentList(), metrics(),
                                            viewport->getSwapchainSize());
    wgpuRenderPassEncoderEnd(renderPassEncoder);
    wgpuRenderPassEncoderRelease(renderPassEncoder);

    return result;
}

Result Implementation::underlyingBegin() {
    JST_CHECK(viewport->pollEvents());

    framebufferTexture = nullptr;

    auto device = Backend::State<DeviceType::WebGPU>()->getDevice();
    WGPUCommandEncoderDescriptor encDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
    if (!encoder) {
        return Result::ERROR;
    }

    for (auto& surface : surfaces) {
        const Result prepareResult = surface->prepare();
        if (prepareResult != Result::SUCCESS && prepareResult != Result::RELOAD) {
            wgpuCommandEncoderRelease(encoder);
            encoder = nullptr;
            return prepareResult;
        }
    }

    this->updateMetrics(*viewport);
    this->updateInput(*viewport);

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
        const auto& framebuffer = *surface->getConfig().framebuffer;
        result = surface->draw(encoder, textureVisibleForPresentation(framebuffer));
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
    }

    auto pass = beginPresent(transfers);
    if (pass.presenting) {
        result = viewport->nextDrawable();
        if (result == Result::SUCCESS) {
            result = viewport->commitDrawable(&framebufferTexture);
        }
        if (result == Result::RECREATE) {
            acquired(pass, Acquire::Recreate);
        } else if (result == Result::SKIP || (result == Result::SUCCESS && !framebufferTexture)) {
            acquired(pass, Acquire::Dropped);
        } else if (result != Result::SUCCESS) {
            return abortFrame(result);
        }
    }

    if (pass.presenting) {
        result = encodePresent();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return abortFrame(result);
        }
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

    return finishPresent(pass);
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

}  // namespace Jetstream::Render
