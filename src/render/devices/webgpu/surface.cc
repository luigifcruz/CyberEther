#include "jetstream/render/devices/webgpu/program.hh"
#include "jetstream/render/devices/webgpu/texture.hh"
#include "jetstream/render/devices/webgpu/surface.hh"
#include "jetstream/render/devices/webgpu/kernel.hh"
#include "jetstream/render/devices/webgpu/buffer.hh"

namespace Jetstream::Render {

using Implementation = SurfaceImp<Device::WebGPU>;

Implementation::SurfaceImp(const Config& config) : Surface(config) {
    framebuffer = std::dynamic_pointer_cast<
        TextureImp<Device::WebGPU>>(config.framebuffer);

    for (auto& program : config.programs) {
        programs.push_back(
            std::dynamic_pointer_cast<ProgramImp<Device::WebGPU>>(program)
        );
    }

    for (auto& kernel : config.kernels) {
        kernels.push_back(
            std::dynamic_pointer_cast<KernelImp<Device::WebGPU>>(kernel)
        );
    }
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating surface.");

    JST_CHECK(createFramebuffer());

    for (auto& program : programs) {
        JST_CHECK(program->create(framebuffer->getTextureFormat()));
    }

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->create());
    }

    requestedSize = framebuffer->size();

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying surface.");

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->destroy());
    }

    for (auto& program : programs) {
        JST_CHECK(program->destroy());
    }

    JST_CHECK(destroyFramebuffer());

    return Result::SUCCESS;
}

Result Implementation::createFramebuffer() {
    JST_DEBUG("[WebGPU] Creating surface framebuffer.");

    return framebuffer->create();
}

Result Implementation::destroyFramebuffer() {
    JST_DEBUG("[WebGPU] Destroying surface framebuffer");

    return framebuffer->destroy();
}

Result Implementation::draw(WGPUCommandEncoder& commandEncoder) {
    if (framebuffer->size(requestedSize)) {
        JST_CHECK(destroyFramebuffer());
        JST_CHECK(createFramebuffer());
    }

    // Encode kernels.

    WGPUComputePassEncoder computePassEncoder =
        wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->encode(computePassEncoder));
    }

    wgpuComputePassEncoderEnd(computePassEncoder);

    // Begin render pass.

    WGPURenderPassColorAttachment colorAttachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    colorAttachment.view = framebuffer->getViewHandle();
    colorAttachment.loadOp = WGPULoadOp_Clear;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue.r = config.clearColor.r;
    colorAttachment.clearValue.g = config.clearColor.g;
    colorAttachment.clearValue.b = config.clearColor.b;
    colorAttachment.clearValue.a = config.clearColor.a;

    WGPURenderPassDescriptor renderPass = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    renderPass.colorAttachmentCount = 1;
    renderPass.colorAttachments = &colorAttachment;
    renderPass.depthStencilAttachment = nullptr;

    // Encode programs.

    WGPURenderPassEncoder renderPassEncoder =
        wgpuCommandEncoderBeginRenderPass(commandEncoder, &renderPass);

    for (auto& program : programs) {
        JST_CHECK(program->draw(renderPassEncoder));
    }

    wgpuRenderPassEncoderEnd(renderPassEncoder);

    return Result::SUCCESS;
}

const Extent2D<U64>& Implementation::size(const Extent2D<U64>& size) {
    if (!framebuffer) {
        return NullSize2D;
    }

    requestedSize = size;

    return framebuffer->size();
}

}  // namespace Jetstream::Render
