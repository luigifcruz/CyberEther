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

    CHECK(_createFramebuffer());

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Surface::destroy() {
    for (auto &program : programs) {
        CHECK(program->destroy());
    }

    CHECK(_destroyFramebuffer());

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Size2D<int> Metal::Surface::size(const Size2D<int>& size) {
    if (!framebuffer) {
        return {-1, -1};
    }

    if (framebuffer->size(size)) {
        RENDER_CHECK_THROW(_destroyFramebuffer());
        RENDER_CHECK_THROW(_createFramebuffer());
    }

    return framebuffer->size();
}

Result Metal::Surface::draw() {
    /*
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    if (framebuffer) {
        auto [width, height] = framebuffer->size();
        glViewport(0, 0, width, height);
    }
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    for (auto &program : programs) {
        CHECK(program->draw());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Surface::_createFramebuffer() {
    /*
    if (framebuffer) {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        CHECK(framebuffer->create());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebuffer->raw(), 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    */
    return Result::SUCCESS;
}

Result Metal::Surface::_destroyFramebuffer() {
    /*
    if (framebuffer) {
        CHECK(framebuffer->destroy());
        glDeleteFramebuffers(1, &fbo);
    }
    */
    return Result::SUCCESS;
}

} // namespace Render
