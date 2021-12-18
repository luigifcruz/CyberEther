#include "render/metal/surface.hpp"
#include "render/metal/program.hpp"
#include "render/metal/texture.hpp"

namespace Render {

Result Metal::Surface::create() {
    framebuffer = std::dynamic_pointer_cast<Metal::Texture>(cfg.framebuffer);

    for (const auto& program : cfg.programs) {
        programs.push_back(std::dynamic_pointer_cast<Metal::Program>(program));
    }

    for (auto &program : programs) {
        CHECK(program->create());
    }

    renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();
    assert(renderPassDesc);

    framebuffer->create();

    auto colorAttachDescOff = renderPassDesc->colorAttachments()->object(0);
    colorAttachDescOff->setTexture(framebuffer->getTexture());
    colorAttachDescOff->setLoadAction(MTL::LoadActionClear);
    colorAttachDescOff->setStoreAction(MTL::StoreActionStore);
    colorAttachDescOff->setClearColor(MTL::ClearColor(0, 0, 0, 0));

    fmt::print("surface ok!\n");

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Surface::destroy() {
    for (auto &program : programs) {
        CHECK(program->destroy());
    }

    framebuffer->destroy();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Size2D<int> Metal::Surface::size(const Size2D<int>& size) {
    if (!framebuffer) {
        return {-1, -1};
    }

    if (framebuffer->size(size)) {
        // TODO: Implement resize.
    }

    return framebuffer->size();
}

Result Metal::Surface::draw(MTL::CommandBuffer* commandBuffer) {
    for (auto &program : programs) {
        CHECK(program->draw(commandBuffer, renderPassDesc));
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
