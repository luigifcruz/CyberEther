#include <algorithm>
#include <limits>
#include <span>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/curves.hh"

#include "jetstream/types.hh"
#include "resources/shaders/global_shaders.hh"

namespace Jetstream::Render::Components {

static_assert(sizeof(Curves::Instance) == 16 * sizeof(F32));

struct Curves::Impl {
    struct UniformBuffer {
        glm::vec4 frame;
        glm::vec4 params;
    };

    UniformBuffer uniforms;

    TypedTensor<Extent2D<F32>> templateVertices;
    TypedTensor<U32> templateIndices;
    TypedTensor<Instance> instanceData;

    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Buffer> verticesBuffer;
    std::shared_ptr<Render::Buffer> indicesBuffer;
    std::shared_ptr<Render::Buffer> instancesBuffer;

    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;

    U64 instanceCount = 0;
    U64 appliedInstanceCount = 0;
    bool enabled = true;

    bool updateUniformBufferFlag = false;
    bool updateInstancesFlag = false;

    void applyEnabled() {
        if (program) {
            program->setEnabled(enabled && instanceCount > 0);
        }
    }
};

Curves::Curves(const Config& config) {
    this->config = config;
    this->config.capacity = std::max<U64>(1, this->config.capacity);
    this->config.maxSegments = std::max<U64>(1, this->config.maxSegments);
    this->pimpl = std::make_unique<Impl>();
}

Curves::~Curves() {
    pimpl.reset();
}

Result Curves::create(Window* window) {
    constexpr U64 maxCount = std::numeric_limits<U32>::max();
    constexpr U64 maxBytes = std::numeric_limits<std::ptrdiff_t>::max();
    constexpr U64 maxExactPoint = U64{1} << std::numeric_limits<F32>::digits;
    if (config.maxSegments > maxCount / 18 ||
        config.maxSegments > maxCount / 4 - 1 ||
        config.maxSegments > maxBytes / sizeof(Extent2D<F32>) / 4 - 1 ||
        config.maxSegments > maxBytes / sizeof(U32) / 18 ||
        config.maxSegments > maxExactPoint ||
        config.capacity > maxCount || config.capacity > maxBytes / sizeof(Instance)) {
        JST_ERROR("[CURVES] Capacity exceeds allocation or draw limits.");
        return Result::ERROR;
    }

    const U64 points = config.maxSegments + 1;
    const U64 vertexCount = points * 4;
    const U64 indexCount = config.maxSegments * 18;

    JST_CHECK(pimpl->templateVertices.create(DeviceType::CPU, {vertexCount}));
    JST_CHECK(pimpl->templateIndices.create(DeviceType::CPU, {indexCount}));
    JST_CHECK(pimpl->instanceData.create(DeviceType::CPU, {config.capacity}));

    constexpr F32 sides[4] = {2.0f, 1.0f, -1.0f, -2.0f};
    for (U64 point = 0; point < points; ++point) {
        for (U64 side = 0; side < 4; ++side) {
            pimpl->templateVertices.data()[point * 4 + side] = {static_cast<F32>(point), sides[side]};
        }
    }

    U32* indices = pimpl->templateIndices.data();
    for (U64 segment = 0; segment < config.maxSegments; ++segment) {
        const U32 a = static_cast<U32>(segment * 4);
        const U32 b = a + 4;
        const U32 triangles[18] = {
            b + 1, a + 1, a + 2,
            a + 2, b + 2, b + 1,
            b + 1, a + 1, a + 0,
            a + 0, b + 0, b + 1,
            b + 2, a + 2, a + 3,
            a + 3, b + 3, b + 2,
        };
        std::copy_n(triangles, 18, indices + segment * 18);
    }

    std::fill_n(pimpl->instanceData.data(), config.capacity, Instance{});

    pimpl->uniforms.frame = glm::vec4(config.framebufferSize.x, config.framebufferSize.y, 0.0f, 0.0f);
    pimpl->uniforms.params = glm::vec4(1.0f, 1.0f, static_cast<F32>(config.maxSegments), 0.0f);

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
        cfg.buffer = pimpl->templateVertices.data();
        cfg.elementByteSize = sizeof(glm::vec2);
        cfg.size = vertexCount;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->verticesBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->templateIndices.data();
        cfg.elementByteSize = sizeof(U32);
        cfg.size = indexCount;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(pimpl->indicesBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = pimpl->instanceData.data();
        cfg.elementByteSize = sizeof(Instance);
        cfg.size = config.capacity;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->instancesBuffer, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {pimpl->verticesBuffer, 2},
        };
        cfg.instances = {
            {pimpl->instancesBuffer, sizeof(Instance) / sizeof(F32)},
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
        pimpl->appliedInstanceCount = 0;
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = GlobalShadersPackage["curves"];
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
    pimpl->updateInstancesFlag = true;
    pimpl->applyEnabled();

    return Result::SUCCESS;
}

Result Curves::surface(Render::Surface::Config& config) {
    config.programs.push_back(pimpl->program);
    return Result::SUCCESS;
}

std::span<Curves::Instance> Curves::instances() {
    return {pimpl->instanceData.data(), static_cast<std::size_t>(config.capacity)};
}

Result Curves::update(U64 instanceCount) {
    pimpl->instanceCount = std::min(instanceCount, config.capacity);
    pimpl->updateInstancesFlag = true;
    pimpl->applyEnabled();
    return Result::SUCCESS;
}

Result Curves::updateTransform(const Extent2D<F32>& framebufferSize,
                               const Extent2D<F32>& origin,
                               F32 scale,
                               F32 fringe) {
    const Impl::UniformBuffer next = {
        .frame = glm::vec4(framebufferSize.x, framebufferSize.y, origin.x, origin.y),
        .params = glm::vec4(scale, fringe, static_cast<F32>(config.maxSegments), 0.0f),
    };
    if (next.frame == pimpl->uniforms.frame && next.params == pimpl->uniforms.params) {
        return Result::SUCCESS;
    }
    config.framebufferSize = framebufferSize;
    pimpl->uniforms = next;
    pimpl->updateUniformBufferFlag = true;
    return Result::SUCCESS;
}

Result Curves::updateScissorRect(const std::optional<Render::ScissorRect>& rect) {
    pimpl->program->scissorRect(rect);
    return Result::SUCCESS;
}

void Curves::enabled(bool value) {
    pimpl->enabled = value;
    pimpl->applyEnabled();
}

Result Curves::present() {
    if (pimpl->updateUniformBufferFlag) {
        JST_CHECK(pimpl->uniformBuffer->update());
        pimpl->updateUniformBufferFlag = false;
    }

    if (pimpl->updateInstancesFlag) {
        if (pimpl->instanceCount > 0) {
            JST_CHECK(pimpl->instancesBuffer->update(0, pimpl->instanceCount));
            if (pimpl->appliedInstanceCount != pimpl->instanceCount) {
                JST_CHECK(pimpl->draw->updateInstanceCount(pimpl->instanceCount));
                pimpl->appliedInstanceCount = pimpl->instanceCount;
            }
        }
        pimpl->updateInstancesFlag = false;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
