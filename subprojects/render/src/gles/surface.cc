#include "gles/surface.hpp"

namespace Render {

Result GLES::Surface::create() {
    if (!cfg.texture) {
        fbo = 0;
        return Result::SUCCESS;
    }

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    ASSERT_SUCCESS(cfg.texture->create());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cfg.texture->raw(), 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::destroy() {
    if (!cfg.texture) {
        return Result::SUCCESS;
    }

    ASSERT_SUCCESS(cfg.texture->destroy());
    glDeleteFramebuffers(1, &fbo);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::start() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, *cfg.width, *cfg.height);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::end() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
