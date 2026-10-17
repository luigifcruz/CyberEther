#include <array>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/image.hh"

#include "jetstream/types.hh"
#include "resources/shaders/global_shaders.hh"

namespace Jetstream::Render::Components {

struct Image::Impl {
    struct UniformBuffer {
        glm::vec4 frame;
        glm::vec4 rect;
        glm::vec4 params;
    };

    UniformBuffer uniforms;

    std::array<Extent2D<F32>, 4> vertexPositions = {{
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f},
    }};
    std::array<U32, 6> vertexIndices = {0, 1, 2, 0, 2, 3};

    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Buffer> positionsBuffer;
    std::shared_ptr<Render::Buffer> indicesBuffer;

    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;

    bool enabled = true;
    bool updateUniformBufferFlag = false;
};

Image::Image(const Config& config) {
    this->config = config;
    this->pimpl = std::make_unique<Impl>();
}

Image::~Image() {
    pimpl.reset();
}

Result Image::create(Window* window) {
    if (config.punch && !window->supportsEraseBlending()) {
        JST_ERROR("[IMAGE] Erase blending is not supported.");
        return Result::ERROR;
    }
    if (!config.punch && !config.texture) {
        JST_ERROR("[IMAGE] Texture is required.");
        return Result::ERROR;
    }

    pimpl->uniforms.frame = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f);
    pimpl->uniforms.rect = glm::vec4(0.0f);
    pimpl->uniforms.params = glm::vec4(0.0f, 1.0f,
                                     config.sourceAlpha == SourceAlpha::PREMULTIPLIED ? 1.0f : 0.0f, 0.0f);

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &pimpl->uniforms;
        cfg.elementByteSize = sizeof(pimpl->uniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(pimpl->uniformBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->vertexPositions.data();
        cfg.elementByteSize = sizeof(glm::vec2);
        cfg.size = pimpl->vertexPositions.size();
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->positionsBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->vertexIndices.data();
        cfg.elementByteSize = sizeof(U32);
        cfg.size = pimpl->vertexIndices.size();
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(pimpl->indicesBuffer, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {pimpl->positionsBuffer, 2},
        };
        cfg.indices = pimpl->indicesBuffer;
        JST_CHECK(window->build(pimpl->vertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.numberOfDraws = 1;
        cfg.numberOfInstances = 1;
        cfg.buffer = pimpl->vertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(pimpl->draw, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = GlobalShadersPackage[config.punch ? "punch" : "image"];
        cfg.draws = {
            pimpl->draw,
        };
        if (!config.punch) {
            cfg.textures = {
                config.texture,
            };
        }
        cfg.buffers = {
            {pimpl->uniformBuffer, Render::Program::Target::VERTEX |
                                   Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = !config.punch;
        cfg.eraseBlending = config.punch;
        cfg.borrowedTextures = !config.punch;
        JST_CHECK(window->build(pimpl->program, cfg));
    }

    pimpl->updateUniformBufferFlag = true;
    pimpl->program->setEnabled(pimpl->enabled);

    return Result::SUCCESS;
}

Result Image::destroy(Window*) {
    return Result::SUCCESS;
}

Result Image::surface(Render::Surface::Config& config) {
    config.programs.push_back(pimpl->program);
    return Result::SUCCESS;
}

Result Image::update(const Extent2D<F32>& framebufferSize,
                     const Extent2D<F32>& translation,
                     const Rect& rect,
                     const F32 cornerRadius,
                     const F32 opacity) {
    const glm::vec4 frame(framebufferSize.x, framebufferSize.y, translation.x, translation.y);
    const glm::vec4 bounds(rect.x, rect.y, rect.width, rect.height);
    const glm::vec4 params(cornerRadius, opacity,
                          config.sourceAlpha == SourceAlpha::PREMULTIPLIED ? 1.0f : 0.0f, 0.0f);
    if (pimpl->uniforms.frame == frame && pimpl->uniforms.rect == bounds && pimpl->uniforms.params == params) {
        return Result::SUCCESS;
    }
    pimpl->uniforms.frame = frame;
    pimpl->uniforms.rect = bounds;
    pimpl->uniforms.params = params;
    pimpl->updateUniformBufferFlag = true;
    return Result::SUCCESS;
}

Result Image::updateScissorRect(const std::optional<Render::ScissorRect>& rect) {
    pimpl->program->scissorRect(rect);
    return Result::SUCCESS;
}

void Image::enabled(bool value) {
    pimpl->enabled = value;
    if (pimpl->program) {
        pimpl->program->setEnabled(value);
    }
}

Result Image::present() {
    if (pimpl->updateUniformBufferFlag) {
        JST_CHECK(pimpl->uniformBuffer->update());
        pimpl->updateUniformBufferFlag = false;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
