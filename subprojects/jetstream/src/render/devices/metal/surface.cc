#include "jetstream/render/metal/program.hh"
#include "jetstream/render/metal/texture.hh"
#include "jetstream/render/metal/surface.hh"

namespace Jetstream::Render {

using Implementation = SurfaceImp<Device::Metal>;

Implementation::SurfaceImp(const Config& config) : Surface(config) {
    framebuffer = std::dynamic_pointer_cast<
        TextureImp<Device::Metal>>(config.framebuffer);

    for (auto& program : config.programs) {
        programs.push_back(
            std::dynamic_pointer_cast<ProgramImp<Device::Metal>>(program)
        );
    }
}

const Result Implementation::create() {
    JST_DEBUG("Creating Metal surface.");

    renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    JST_ASSERT(renderPassDescriptor);

    JST_CHECK(createFramebuffer());

    for (auto& program : programs) {
        JST_CHECK(program->create(framebuffer->getPixelFormat()));
    }

    return Result::SUCCESS;
}

const Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal surface.");

    for (auto& program : programs) {
        JST_CHECK(program->destroy());
    }

    JST_CHECK(destroyFramebuffer());

    renderPassDescriptor->release();

    return Result::SUCCESS;
}

const Result Implementation::createFramebuffer() {
    JST_DEBUG("Creating Metal surface framebuffer.");

    JST_CHECK(framebuffer->create());

    auto colorAttachDescOff = renderPassDescriptor->colorAttachments()->object(0);
    auto texture = reinterpret_cast<const MTL::Texture*>(framebuffer->raw());
    colorAttachDescOff->setTexture(texture);
    colorAttachDescOff->setLoadAction(MTL::LoadActionClear);
    colorAttachDescOff->setStoreAction(MTL::StoreActionStore);
    colorAttachDescOff->setClearColor(MTL::ClearColor(0, 0, 0, 1.0));

    return Result::SUCCESS;
}

const Result Implementation::destroyFramebuffer() {
    JST_DEBUG("Destroying Metal surface framebuffer");

    return framebuffer->destroy();
}

const Result Implementation::draw(MTL::CommandBuffer* commandBuffer) {
    for (auto& program : programs) {
        JST_CHECK(program->draw(commandBuffer, renderPassDescriptor));
    }

    auto blitCommandEncoder = commandBuffer->blitCommandEncoder();
    auto texture = reinterpret_cast<const MTL::Texture*>(framebuffer->raw());
    blitCommandEncoder->synchronizeResource(texture);
    blitCommandEncoder->endEncoding();
    blitCommandEncoder->release();

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
