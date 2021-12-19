#include "render/metal/surface.hpp"
#include "render/metal/program.hpp"
#include "render/metal/texture.hpp"

namespace Render {

Result Metal::Surface::create() {
    framebuffer = std::dynamic_pointer_cast<Metal::Texture>(cfg.framebuffer);

    for (const auto& program : cfg.programs) {
        programs.push_back(std::dynamic_pointer_cast<Metal::Program>(program));
    }

    renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    assert(renderPassDesc);

    CHECK(this->createFramebuffer());

    for (auto &program : programs) {
        CHECK(program->create(framebuffer->getPixelFormat()));
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Surface::destroy() {
    for (auto &program : programs) {
        CHECK(program->destroy());
    }

    CHECK(this->destroyFramebuffer());

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
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

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Surface::createFramebuffer() {
    CHECK(framebuffer->create());

    auto colorAttachDescOff = renderPassDesc->colorAttachments()->object(0);
    colorAttachDescOff->setTexture((MTL::Texture*)framebuffer->raw());
    colorAttachDescOff->setLoadAction(MTL::LoadActionClear);
    colorAttachDescOff->setStoreAction(MTL::StoreActionStore);
    colorAttachDescOff->setClearColor(MTL::ClearColor(0, 0, 0, 0));

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Surface::destroyFramebuffer() {
    return framebuffer->destroy();
}

} // namespace Render
