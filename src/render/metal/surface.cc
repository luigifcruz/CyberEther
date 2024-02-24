#include "jetstream/render/metal/program.hh"
#include "jetstream/render/metal/texture.hh"
#include "jetstream/render/metal/surface.hh"

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
}

Result Implementation::create() {
    JST_DEBUG("Creating Metal surface.");

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor);

    JST_CHECK(createFramebuffer());

    for (auto& program : programs) {
        JST_CHECK(program->create((config.multisampled) ? framebuffer : framebufferResolve));
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal surface.");

    for (auto& program : programs) {
        JST_CHECK(program->destroy());
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

    if (framebufferResolve->size(size)) {
        if (config.multisampled) {
            framebuffer->size(size);
        }

        destroyFramebuffer();
        createFramebuffer();
    }

    return framebufferResolve->size();
} 

}  // namespace Jetstream::Render
