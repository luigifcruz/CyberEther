#include <algorithm>
#include <limits>
#include <span>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/mesh.hh"

#include "jetstream/types.hh"
#include "resources/shaders/global_shaders.hh"

namespace Jetstream::Render::Components {

struct Mesh::Impl {
    struct UniformBuffer {
        glm::vec4 frame;
        glm::vec4 params;
    };

    UniformBuffer uniforms;

    TypedTensor<Extent2D<F32>> vertexPositions;
    TypedTensor<ColorRGBA<F32>> vertexColors;
    TypedTensor<U32> vertexIndices;
    TypedTensor<U32> uploadIndices;

    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Buffer> positionsBuffer;
    std::shared_ptr<Render::Buffer> colorsBuffer;
    std::shared_ptr<Render::Buffer> indicesBuffer;

    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;

    U64 vertexCount = 0;
    U64 indexCount = 0;
    U64 appliedIndexCount = 0;
    bool enabled = true;

    bool updateUniformBufferFlag = false;
    bool updateGeometryFlag = false;

    void applyEnabled() {
        if (program) {
            program->setEnabled(enabled && indexCount > 0);
        }
    }
};

Mesh::Mesh(const Config& config) {
    this->config = config;
    this->config.vertexCapacity = std::max<U64>(3, this->config.vertexCapacity);
    this->config.indexCapacity = std::max<U64>(3, this->config.indexCapacity - this->config.indexCapacity % 3);
    this->pimpl = std::make_unique<Impl>();
}

Mesh::~Mesh() {
    pimpl.reset();
}

Result Mesh::create(Window* window) {
    constexpr U64 maxCount = std::numeric_limits<U32>::max();
    constexpr U64 maxBytes = std::numeric_limits<std::ptrdiff_t>::max();
    if (config.vertexCapacity > maxCount || config.indexCapacity > maxCount ||
        config.vertexCapacity > maxBytes / sizeof(ColorRGBA<F32>) ||
        config.vertexCapacity > maxBytes / sizeof(Extent2D<F32>) ||
        config.indexCapacity > maxBytes / sizeof(U32)) {
        JST_ERROR("[MESH] Capacity exceeds allocation or draw limits.");
        return Result::ERROR;
    }

    JST_CHECK(pimpl->vertexPositions.create(DeviceType::CPU, {config.vertexCapacity}));
    JST_CHECK(pimpl->vertexColors.create(DeviceType::CPU, {config.vertexCapacity}));
    JST_CHECK(pimpl->vertexIndices.create(DeviceType::CPU, {config.indexCapacity}));
    JST_CHECK(pimpl->uploadIndices.create(DeviceType::CPU, {config.indexCapacity}));

    std::fill_n(pimpl->vertexPositions.data(), config.vertexCapacity, Extent2D<F32>{0.0f, 0.0f});
    std::fill_n(pimpl->vertexColors.data(), config.vertexCapacity, ColorRGBA<F32>{0.0f, 0.0f, 0.0f, 0.0f});
    std::fill_n(pimpl->vertexIndices.data(), config.indexCapacity, 0u);
    std::fill_n(pimpl->uploadIndices.data(), config.indexCapacity, 0u);

    pimpl->uniforms.frame = glm::vec4(config.framebufferSize.x, config.framebufferSize.y, 0.0f, 0.0f);
    pimpl->uniforms.params = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);

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
        cfg.size = config.vertexCapacity;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->positionsBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->vertexColors.data();
        cfg.elementByteSize = sizeof(ColorRGBA<F32>);
        cfg.size = config.vertexCapacity;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->colorsBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->uploadIndices.data();
        cfg.elementByteSize = sizeof(U32);
        cfg.size = config.indexCapacity;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(pimpl->indicesBuffer, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {pimpl->positionsBuffer, 2},
            {pimpl->colorsBuffer, 4},
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
        pimpl->appliedIndexCount = 0;
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = GlobalShadersPackage["mesh_color"];
        cfg.draws = {
            pimpl->draw,
        };
        cfg.buffers = {
            {pimpl->uniformBuffer, Render::Program::Target::VERTEX},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(pimpl->program, cfg));
    }

    pimpl->updateUniformBufferFlag = true;
    pimpl->updateGeometryFlag = true;
    pimpl->applyEnabled();

    return Result::SUCCESS;
}

Result Mesh::destroy(Window*) {
    return Result::SUCCESS;
}

Result Mesh::surface(Render::Surface::Config& config) {
    config.programs.push_back(pimpl->program);
    return Result::SUCCESS;
}

std::span<Extent2D<F32>> Mesh::positions() {
    return {pimpl->vertexPositions.data(), static_cast<std::size_t>(config.vertexCapacity)};
}

std::span<ColorRGBA<F32>> Mesh::colors() {
    return {pimpl->vertexColors.data(), static_cast<std::size_t>(config.vertexCapacity)};
}

std::span<U32> Mesh::indices() {
    return {pimpl->vertexIndices.data(), static_cast<std::size_t>(config.indexCapacity)};
}

Result Mesh::update(U64 vertexCount, U64 indexCount) {
    pimpl->vertexCount = std::min(vertexCount, config.vertexCapacity);
    pimpl->indexCount = std::min(indexCount - indexCount % 3, config.indexCapacity);
    pimpl->updateGeometryFlag = true;
    pimpl->applyEnabled();
    return Result::SUCCESS;
}

Result Mesh::updateTransform(const Extent2D<F32>& framebufferSize, const Extent2D<F32>& origin, F32 scale) {
    const glm::vec4 frame(framebufferSize.x, framebufferSize.y, origin.x, origin.y);
    const glm::vec4 params(scale, 0.0f, 0.0f, 0.0f);
    if (pimpl->uniforms.frame == frame && pimpl->uniforms.params == params) {
        return Result::SUCCESS;
    }
    config.framebufferSize = framebufferSize;
    pimpl->uniforms.frame = frame;
    pimpl->uniforms.params = params;
    pimpl->updateUniformBufferFlag = true;
    return Result::SUCCESS;
}

Result Mesh::updateScissorRect(const std::optional<Render::ScissorRect>& rect) {
    pimpl->program->scissorRect(rect);
    return Result::SUCCESS;
}

void Mesh::enabled(bool value) {
    pimpl->enabled = value;
    pimpl->applyEnabled();
}

Result Mesh::present() {
    if (pimpl->updateGeometryFlag && pimpl->indexCount > 0) {
        for (U64 offset = 0; offset < pimpl->indexCount; ++offset) {
            const U32 index = pimpl->vertexIndices.data()[offset];
            if (index >= pimpl->vertexCount) {
                JST_ERROR("[MESH] Index {} at offset {} is outside active vertex count {}.",
                          index, offset, pimpl->vertexCount);
                return Result::ERROR;
            }
        }
    }

    if (pimpl->updateUniformBufferFlag) {
        JST_CHECK(pimpl->uniformBuffer->update());
        pimpl->updateUniformBufferFlag = false;
    }

    if (pimpl->updateGeometryFlag) {
        if (pimpl->vertexCount > 0) {
            JST_CHECK(pimpl->positionsBuffer->update(0, pimpl->vertexCount));
            JST_CHECK(pimpl->colorsBuffer->update(0, pimpl->vertexCount));
        }
        std::copy_n(pimpl->vertexIndices.data(), pimpl->indexCount, pimpl->uploadIndices.data());
        std::fill_n(pimpl->uploadIndices.data() + pimpl->indexCount, config.indexCapacity - pimpl->indexCount, 0u);
        if (pimpl->indexCount > 0) {
            JST_CHECK(pimpl->indicesBuffer->update());
            if (pimpl->appliedIndexCount != pimpl->indexCount) {
                JST_CHECK(pimpl->draw->updateVertexCount(pimpl->indexCount));
                pimpl->appliedIndexCount = pimpl->indexCount;
            }
        }
        pimpl->updateGeometryFlag = false;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
