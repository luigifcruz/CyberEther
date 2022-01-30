#include "render/gles/surface.hpp"
#include "render/gles/program.hpp"
#include "render/gles/texture.hpp"

namespace Render {

GLES::Surface::Surface(const Config& config, const GLES& instance)
         : Render::Surface(config), instance(instance) {
    framebuffer = std::dynamic_pointer_cast<GLES::Texture>(config.framebuffer);

    for (const auto& program : config.programs) {
        programs.push_back(std::dynamic_pointer_cast<GLES::Program>(program));
    }
}

Result GLES::Surface::create() {
    for (auto& program : programs) {
        CHECK(program->create());
    }

    CHECK(createFramebuffer());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::destroy() {
    for (auto& program : programs) {
        CHECK(program->destroy());
    }

    CHECK(destroyFramebuffer());

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Size2D<int> GLES::Surface::size(const Size2D<int>& size) {
    if (!framebuffer) {
        return {-1, -1};
    }

    if (framebuffer->size(size)) {
        RENDER_CHECK_THROW(destroyFramebuffer());
        RENDER_CHECK_THROW(createFramebuffer());
    }

    return framebuffer->size();
}

Result GLES::Surface::draw() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    if (framebuffer) {
        auto [width, height] = framebuffer->size();
        glViewport(0, 0, width, height);
    }
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    for (auto& program : programs) {
        CHECK(program->draw());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return GLES::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result GLES::Surface::createFramebuffer() {
    if (framebuffer) {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        CHECK(framebuffer->create());
        const auto& tex = reinterpret_cast<uintptr_t>(framebuffer->raw());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    return Result::SUCCESS;
}

Result GLES::Surface::destroyFramebuffer() {
    if (framebuffer) {
        CHECK(framebuffer->destroy());
        glDeleteFramebuffers(1, &fbo);
    }
    return Result::SUCCESS;
}

}  // namespace Render
