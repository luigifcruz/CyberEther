#include "gles/surface.hpp"

namespace Render {

Result GLES::Surface::create() {
    if (s.default_s) {
        fbo = 0;
        return Result::SUCCESS;
    }

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s.width, s.height, 0,
            GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // is the GL_COLOR_ATTACHMENT0 global??
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::destroy() {
    if (s.default_s) {
        return Result::SUCCESS;
    }

    glDeleteFramebuffers(1, &fbo);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::start() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::end() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

void* GLES::Surface::getRawTexture() {
    return (void*)tex;
}

} // namespace Render
