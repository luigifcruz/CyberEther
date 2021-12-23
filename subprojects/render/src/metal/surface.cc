#include "render/metal/surface.hpp"
#include "render/metal/program.hpp"
#include "render/metal/texture.hpp"

namespace Render {

Metal::Surface::Surface(const Config& config, const Metal& instance)
         : Render::Surface(config), instance(instance) {
    framebuffer = std::dynamic_pointer_cast<Metal::Texture>(config.framebuffer);

    for (const auto& program : config.programs) {
        programs.push_back(std::dynamic_pointer_cast<Metal::Program>(program));
    }
}

Result Metal::Surface::create() {
    renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    RENDER_ASSERT(renderPassDesc);

    CHECK(this->createFramebuffer());

    for (auto &program : programs) {
        CHECK(program->create(framebuffer->getPixelFormat()));
    }

    return Result::SUCCESS;
}

Result Metal::Surface::destroy() {
    for (auto &program : programs) {
        CHECK(program->destroy());
    }

    CHECK(this->destroyFramebuffer());

    renderPassDesc->release();

    return Result::SUCCESS;
}

Size2D<int> Metal::Surface::size(const Size2D<int>& size) {
    if (!framebuffer) {
        return {-1, -1};
    }

    if (framebuffer->size(size)) {
        this->destroyFramebuffer();
        this->createFramebuffer();
    }

    return framebuffer->size();
}

Result Metal::Surface::draw(MTL::CommandBuffer* commandBuffer) {
    for (auto &program : programs) {
        CHECK(program->draw(commandBuffer, renderPassDesc));
    }

    auto blitCommandEncoder = commandBuffer->blitCommandEncoder();
    blitCommandEncoder->synchronizeResource((MTL::Texture*)framebuffer->raw());
    blitCommandEncoder->endEncoding();

    return Result::SUCCESS;
}

Result Metal::Surface::createFramebuffer() {
    CHECK(framebuffer->create());

    auto colorAttachDescOff = renderPassDesc->colorAttachments()->object(0);
    colorAttachDescOff->setTexture((MTL::Texture*)framebuffer->raw());
    colorAttachDescOff->setLoadAction(MTL::LoadActionClear);
    colorAttachDescOff->setStoreAction(MTL::StoreActionStore);
    colorAttachDescOff->setClearColor(MTL::ClearColor(0, 0, 0, 0));

    return Result::SUCCESS;
}

Result Metal::Surface::destroyFramebuffer() {
    return framebuffer->destroy();
}

}  // namespace Render
