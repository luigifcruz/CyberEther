#include "jetstream/render/metal/program.hh"
#include "jetstream/render/metal/texture.hh"
#include "jetstream/render/metal/surface.hh"
#include "jetstream/render/metal/kernel.hh"
#include "jetstream/render/metal/buffer.hh"

namespace Jetstream::Render {

using Implementation = SurfaceImp<Device::Metal>;

Implementation::SurfaceImp(const Config& config) : Surface(config) {
    framebufferResolve = std::dynamic_pointer_cast<
        TextureImp<Device::Metal>>(config.framebuffer);

    if (config.multisampled) {
        auto framebuffer_config = framebufferResolve->getConfig();
        framebuffer_config.multisampled = true;
        framebuffer = std::make_shared<TextureImp<Device::Metal>>(framebuffer_config);
    }

    for (auto& program : config.programs) {
        programs.push_back(
            std::dynamic_pointer_cast<ProgramImp<Device::Metal>>(program)
        );
    }

    for (auto& kernel : config.kernels) {
        kernels.push_back(
            std::dynamic_pointer_cast<KernelImp<Device::Metal>>(kernel)
        );
    }

    for (auto& buffer : config.buffers) {
        buffers.push_back(
            std::dynamic_pointer_cast<BufferImp<Device::Metal>>(buffer)
        );
    }
}

Result Implementation::create() {
    JST_DEBUG("Creating Metal surface.");

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor);

    JST_CHECK(createFramebuffer());

    for (auto& buffer : buffers) {
        JST_CHECK(buffer->create());
    }

    for (auto& program : programs) {
        JST_CHECK(program->create((config.multisampled) ? framebuffer : framebufferResolve));
    }

    for (auto& kernel : kernels) {
        JST_CHECK(kernel->create());
    }

    requestedSize = framebufferResolve->size();

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal surface.");

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

    renderPassDescriptor->release();

    return Result::SUCCESS;
}

Result Implementation::createFramebuffer() {
    JST_DEBUG("Creating Metal surface framebuffer.");

    JST_CHECK(framebufferResolve->create());
    if (config.multisampled) {
        JST_CHECK(framebuffer->create());
    }

    auto colorAttachDescOff = renderPassDescriptor->colorAttachments()->object(0)->init();

    if (config.multisampled) {
        auto textureResolve = reinterpret_cast<const MTL::Texture*>(framebufferResolve->raw());
        auto texture = reinterpret_cast<const MTL::Texture*>(framebuffer->raw());
        colorAttachDescOff->setTexture(texture);
        colorAttachDescOff->setResolveTexture(textureResolve);
        colorAttachDescOff->setLoadAction(MTL::LoadActionClear);
        colorAttachDescOff->setStoreAction(MTL::StoreActionMultisampleResolve);
        colorAttachDescOff->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 0.0));
    } else {
        auto texture = reinterpret_cast<const MTL::Texture*>(framebufferResolve->raw());
        colorAttachDescOff->setTexture(texture);
        colorAttachDescOff->setLoadAction(MTL::LoadActionClear);
        colorAttachDescOff->setStoreAction(MTL::StoreActionStore);
        colorAttachDescOff->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 0.0));
    }

    return Result::SUCCESS;
}

Result Implementation::destroyFramebuffer() {
    JST_DEBUG("Destroying Metal surface framebuffer");

    if (config.multisampled) {
        JST_CHECK(framebuffer->destroy());
    }
    JST_CHECK(framebufferResolve->destroy());

    return Result::SUCCESS;
}

Result Implementation::draw(MTL::CommandBuffer* commandBuffer) {
    if (framebufferResolve->size(requestedSize)) {
        if (config.multisampled) {
            framebuffer->size(requestedSize);
        }

        JST_CHECK(destroyFramebuffer());
        JST_CHECK(createFramebuffer());
    }

    // Encode kernels.

    auto computeCmdEncoder = commandBuffer->computeCommandEncoder();
    for (auto& kernel : kernels) {
        JST_CHECK(kernel->encode(computeCmdEncoder));
    }
    computeCmdEncoder->endEncoding();

    // Encode programs.

    auto renderCmdEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor);
    for (auto& program : programs) {
        JST_CHECK(program->draw(renderCmdEncoder));
    }
    renderCmdEncoder->endEncoding();

    return Result::SUCCESS;
}

const Size2D<U64>& Implementation::size(const Size2D<U64>& size) { 
    if (!framebufferResolve) {
        return NullSize;
    }

    requestedSize = size;

    return framebufferResolve->size();
} 

}  // namespace Jetstream::Render
