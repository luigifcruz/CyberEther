#include <algorithm>
#include <array>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/orb.hh"

#include "jetstream/types.hh"
#include "resources/shaders/global_shaders.hh"

namespace Jetstream::Render::Components {

namespace {

constexpr F32 kOrbExtent = 0.6f;

glm::vec4 Color(const ColorRGBA<F32>& color) {
    return glm::vec4(color.r, color.g, color.b, color.a);
}

}  // namespace

struct Orb::Impl {
    struct UniformBuffer {
        glm::vec4 rect;
        glm::vec4 orb;
        glm::vec4 params;
        glm::vec4 listeningColor;
        glm::vec4 busyColor;
        glm::vec4 listeningAccent;
        glm::vec4 busyAccent;

        bool operator==(const UniformBuffer&) const = default;
    };

    UniformBuffer uniforms{};

    std::array<Extent2D<F32>, 4> vertexPositions = {{
        {-1.0f, -1.0f},
        {1.0f, -1.0f},
        {1.0f, 1.0f},
        {-1.0f, 1.0f},
    }};
    std::array<U32, 6> vertexIndices = {0, 1, 2, 2, 3, 0};

    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Buffer> positionsBuffer;
    std::shared_ptr<Render::Buffer> indicesBuffer;

    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;

    bool enabled = true;
    bool drawable = false;
    bool updateUniformBufferFlag = false;

    void applyEnabled() {
        if (program) {
            program->setEnabled(enabled && drawable);
        }
    }
};

Orb::Orb(const Config& config) {
    this->config = config;
    this->pimpl = std::make_unique<Impl>();
}

Orb::~Orb() {
    pimpl.reset();
}

Result Orb::create(Window* window) {
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
        cfg.shaders = GlobalShadersPackage["orb"];
        cfg.draws = {
            pimpl->draw,
        };
        cfg.buffers = {
            {pimpl->uniformBuffer, Render::Program::Target::VERTEX |
                                   Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(pimpl->program, cfg));
    }

    pimpl->updateUniformBufferFlag = true;
    pimpl->applyEnabled();

    return Result::SUCCESS;
}

Result Orb::surface(Render::Surface::Config& config) {
    config.programs.push_back(pimpl->program);
    return Result::SUCCESS;
}

Result Orb::update(const Extent2D<F32>& framebufferSize, const State& state) {
    config.framebufferSize = framebufferSize;
    const Rect& rect = state.rect;
    pimpl->drawable = !rect.empty() && state.radius > 0.0f && framebufferSize.x > 0.0f && framebufferSize.y > 0.0f;
    pimpl->applyEnabled();
    if (!pimpl->drawable) {
        return Result::SUCCESS;
    }

    const Extent2D<F32> middle = rect.center();
    const F32 halfWidth = std::max(rect.width * 0.5f, 1e-3f);
    const F32 halfHeight = std::max(rect.height * 0.5f, 1e-3f);
    const F32 unit = state.radius / kOrbExtent;

    const Impl::UniformBuffer next = {
        .rect = glm::vec4(-1.0f + 2.0f * middle.x / framebufferSize.x,
                          1.0f - 2.0f * middle.y / framebufferSize.y,
                          rect.width / framebufferSize.x,
                          -rect.height / framebufferSize.y),
        .orb = glm::vec4((state.center.x - middle.x) / halfWidth,
                         (state.center.y - middle.y) / halfHeight,
                         rect.width * 0.5f / unit,
                         rect.height * 0.5f / unit),
        .params = glm::vec4(state.time, state.busy, state.activity, state.level),
        .listeningColor = Color(state.listeningColor),
        .busyColor = Color(state.busyColor),
        .listeningAccent = Color(state.listeningAccent),
        .busyAccent = Color(state.busyAccent),
    };
    if (next == pimpl->uniforms) {
        return Result::SUCCESS;
    }
    pimpl->uniforms = next;
    pimpl->updateUniformBufferFlag = true;
    return Result::SUCCESS;
}

Result Orb::updateScissorRect(const std::optional<Render::ScissorRect>& rect) {
    pimpl->program->scissorRect(rect);
    return Result::SUCCESS;
}

void Orb::enabled(bool value) {
    pimpl->enabled = value;
    pimpl->applyEnabled();
}

Result Orb::present() {
    if (pimpl->updateUniformBufferFlag) {
        JST_CHECK(pimpl->uniformBuffer->update());
        pimpl->updateUniformBufferFlag = false;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
