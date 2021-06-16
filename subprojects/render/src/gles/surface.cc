#include "render/gles/surface.hpp"
#include "render/gles/program.hpp"
#include "render/gles/texture.hpp"

namespace Render {

Result GLES::Surface::create() {
    framebuffer = std::dynamic_pointer_cast<GLES::Texture>(cfg.framebuffer);

    for (const auto& program : cfg.programs) {
        programs.push_back(std::dynamic_pointer_cast<GLES::Program>(program));
    }

    for (auto &program : programs) {
        CHECK(program->create());
    }

    CHECK(_createFramebuffer());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::destroy() {
    for (auto &program : programs) {
        CHECK(program->destroy());
    }

    CHECK(_destroyFramebuffer());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::resize(int w, int h) {
    if (w <= 1 || h <= 1) {
        return Result::ERROR;
    }

    framebuffer->cfg.width = w;
    framebuffer->cfg.height = h;

    CHECK(_destroyFramebuffer());
    CHECK(_createFramebuffer());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::draw() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, framebuffer->cfg.width, framebuffer->cfg.height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    for (auto &program : programs) {
        CHECK(program->draw());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::_createFramebuffer() {
    if (framebuffer) {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        CHECK(framebuffer->create());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebuffer->raw(), 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    return Result::SUCCESS;
}

Result GLES::Surface::_destroyFramebuffer() {
    if (framebuffer) {
        CHECK(framebuffer->destroy());
        glDeleteFramebuffers(1, &fbo);
    }
    return Result::SUCCESS;
}

} // namespace Render
