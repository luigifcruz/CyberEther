#include "jetstream/render/webgpu/program.hh"
#include "jetstream/render/webgpu/texture.hh"
#include "jetstream/render/webgpu/surface.hh"

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
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating surface.");

    JST_CHECK(createFramebuffer());

    for (auto& program : programs) {
        JST_CHECK(program->create(framebuffer->getTextureFormat()));
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying surface.");

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

Result Implementation::draw(wgpu::CommandEncoder& commandEncoder) {
    wgpu::RenderPassColorAttachment colorAttachment{};
    colorAttachment.view = framebuffer->getViewHandle();
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.storeOp = wgpu::StoreOp::Store;
    colorAttachment.clearValue.r = 0.0f;
    colorAttachment.clearValue.g = 0.0f;
    colorAttachment.clearValue.b = 0.0f;
    colorAttachment.clearValue.a = 1.0f;

    wgpu::RenderPassDescriptor renderPass{};
    renderPass.colorAttachmentCount = 1;
    renderPass.colorAttachments = &colorAttachment;
    renderPass.depthStencilAttachment = nullptr;

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

    if (framebuffer->size(size)) {
        destroyFramebuffer();
        createFramebuffer();
    }

    return framebuffer->size();
} 

}  // namespace Jetstream::Render
