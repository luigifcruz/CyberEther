#include "render/gles/surface.hpp"

namespace Render {

Result GLES::Surface::create() {
    for (auto &program : cfg.programs) {
        ASSERT_SUCCESS(program->create());
    }

    if (cfg.texture) {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        ASSERT_SUCCESS(cfg.texture->create());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cfg.texture->raw(), 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::destroy() {
    for (auto &program : cfg.programs) {
        ASSERT_SUCCESS(program->destroy());
    }

    if (cfg.texture) {
        ASSERT_SUCCESS(cfg.texture->destroy());
        glDeleteFramebuffers(1, &fbo);
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::draw() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, *cfg.width, *cfg.height);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    for (auto &program : cfg.programs) {
        ASSERT_SUCCESS(program->draw());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
