#include "jetstream/render/webgpu/program.hh"
#include "jetstream/render/webgpu/texture.hh"
#include "jetstream/render/webgpu/surface.hh"
#include "jetstream/render/webgpu/kernel.hh"
#include "jetstream/render/webgpu/buffer.hh"

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

    for (auto& buffer : config.buffers) {
        buffers.push_back(
            std::dynamic_pointer_cast<BufferImp<Device::WebGPU>>(buffer)
        );
    }
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating surface.");

    JST_CHECK(createFramebuffer());

    for (auto& buffer : buffers) {
        JST_CHECK(buffer->create());
    }

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

    for (auto& buffer : buffers) {
        JST_CHECK(buffer->destroy());
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

Result Implementation::draw(wgpu::CommandEncoder& commandEncoder) {
    if (framebuffer->size(requestedSize)) {
        JST_CHECK(destroyFramebuffer());
        JST_CHECK(createFramebuffer());
    }

    // Encode kernels.

    auto computePassEncoder = commandEncoder.BeginComputePass();

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->encode(computePassEncoder));
    }

    computePassEncoder.End();

    // Begin render pass.

    wgpu::RenderPassColorAttachment colorAttachment{};
    colorAttachment.view = framebuffer->getViewHandle();
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.storeOp = wgpu::StoreOp::Store;
    colorAttachment.clearValue.r = 0.0f;
    colorAttachment.clearValue.g = 0.0f;
    colorAttachment.clearValue.b = 0.0f;
    colorAttachment.clearValue.a = 0.0f;

    wgpu::RenderPassDescriptor renderPass{};
    renderPass.colorAttachmentCount = 1;
    renderPass.colorAttachments = &colorAttachment;
    renderPass.depthStencilAttachment = nullptr;

    // Encode programs.

    auto renderPassEncoder = commandEncoder.BeginRenderPass(&renderPass);

    for (auto& program : programs) {
        JST_CHECK(program->draw(renderPassEncoder));
    }

    renderPassEncoder.End();

    return Result::SUCCESS;
}

const Size2D<U64>& Implementation::size(const Size2D<U64>& size) { 
    if (!framebuffer) {
        return NullSize;
    }

    requestedSize = size;

    return framebuffer->size();
} 

}  // namespace Jetstream::Render
