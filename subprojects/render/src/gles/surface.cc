#include "render/gles/surface.hpp"
#include "render/gles/program.hpp"
#include "render/gles/texture.hpp"

namespace Render {

Result GLES::Surface::create() {
    for (auto &program : programs) {
        auto a = static_cast<GLES::Program*>(program.get());
        ASSERT_SUCCESS(a->create());
    }

    if (texture) {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        ASSERT_SUCCESS(texture->create());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture->raw(), 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::destroy() {
    for (auto &program : programs) {
        ASSERT_SUCCESS(program->destroy());
    }

    if (texture) {
        ASSERT_SUCCESS(texture->destroy());
        glDeleteFramebuffers(1, &fbo);
    }

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::draw() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, *texture->cfg.width, *texture->cfg.height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    for (auto &program : programs) {
        ASSERT_SUCCESS(program->draw());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

std::shared_ptr<Texture> GLES::Surface::bind(Render::Texture::Config& cfg) {
    // validation here?
    texture = std::make_shared<GLES::Texture>(cfg, inst);
    return texture;
}

std::shared_ptr<Program> GLES::Surface::bind(Render::Program::Config& cfg) {
    auto program = std::make_shared<GLES::Program>(cfg, inst);
    programs.push_back(program);
    return program;
}

} // namespace Render
