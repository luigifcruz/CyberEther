#include "orb.hh"

#include <jetstream/logger.hh>
#include <jetstream/render/base.hh>

#include "../../context.hh"
#include "../../retained/drawable.hh"
#include "../../retained/helpers.hh"

#include "resources/shaders/global_shaders.hh"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace Jetstream::Sakura::Retained {

namespace {

struct Uniforms {
    F32 rect[4];
    F32 orb[4];
    F32 params[4];
    F32 colorA[4];
    F32 colorB[4];
    F32 colorC[4];
    F32 colorD[4];
};

void StoreColor(F32* target, const ColorRGBA<F32>& color) {
    target[0] = color.r;
    target[1] = color.g;
    target[2] = color.b;
    target[3] = color.a;
}

}  // namespace

struct OrbView::Impl : public Drawable {
    Config config;
    Context* context = nullptr;
    Uniforms uniforms{};
    std::vector<F32> vertices = {-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
    std::vector<U32> indices = {0, 1, 2, 2, 3, 0};
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Buffer> verticesBuffer;
    std::shared_ptr<Render::Buffer> indicesBuffer;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;

    ~Impl() override {
        if (context && context->release) {
            context->release(this);
        }
    }

    Result attach(Context* context, Render::Surface::Config& surfaceConfig) override {
        this->context = context;

        {
            Render::Buffer::Config cfg;
            cfg.buffer = &uniforms;
            cfg.elementByteSize = sizeof(uniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(context->render->build(uniformBuffer, cfg));
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = vertices.data();
            cfg.elementByteSize = sizeof(F32) * 2;
            cfg.size = vertices.size() / 2;
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(context->render->build(verticesBuffer, cfg));
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = indices.data();
            cfg.elementByteSize = sizeof(U32);
            cfg.size = indices.size();
            cfg.target = Render::Buffer::Target::VERTEX_INDICES;
            JST_CHECK(context->render->build(indicesBuffer, cfg));
        }

        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {verticesBuffer, 2},
            };
            cfg.indices = indicesBuffer;
            JST_CHECK(context->render->build(vertex, cfg));
        }

        {
            Render::Draw::Config cfg;
            cfg.numberOfDraws = 1;
            cfg.numberOfInstances = 1;
            cfg.buffer = vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            JST_CHECK(context->render->build(draw, cfg));
        }

        {
            Render::Program::Config cfg;
            cfg.shaders = GlobalShadersPackage["orb"];
            cfg.draws = {draw};
            cfg.buffers = {
                {uniformBuffer, Render::Program::Target::VERTEX |
                                Render::Program::Target::FRAGMENT},
            };
            cfg.enableAlphaBlending = true;
            JST_CHECK(context->render->build(program, cfg));
        }

        surfaceConfig.programs.push_back(program);
        return Result::SUCCESS;
    }

    Result detach(Render::Window*) override {
        program.reset();
        draw.reset();
        vertex.reset();
        indicesBuffer.reset();
        verticesBuffer.reset();
        uniformBuffer.reset();
        context = nullptr;
        return Result::SUCCESS;
    }

    Result upload() override {
        if (!program || !context) {
            return Result::SUCCESS;
        }

        const bool active = config.visible && !config.rect.empty() && config.orbRadius > 0.0f;
        program->setEnabled(active);
        if (!active) {
            return Result::SUCCESS;
        }

        const auto& framebufferSize = context->framebufferSize;
        const auto center = PixelToNdc(framebufferSize, config.rect.center().x, config.rect.center().y);
        const F32 halfWidth = framebufferSize.x > 0
            ? config.rect.width / static_cast<F32>(framebufferSize.x)
            : 0.0f;
        const F32 halfHeight = framebufferSize.y > 0
            ? config.rect.height / static_cast<F32>(framebufferSize.y)
            : 0.0f;

        uniforms.rect[0] = center.x;
        uniforms.rect[1] = center.y;
        uniforms.rect[2] = halfWidth;
        uniforms.rect[3] = -halfHeight;
        const F32 unit = config.orbRadius / 0.6f;
        uniforms.orb[0] = (config.orbCenter.x - config.rect.center().x) / std::max(config.rect.width * 0.5f, 1e-3f);
        uniforms.orb[1] = (config.orbCenter.y - config.rect.center().y) / std::max(config.rect.height * 0.5f, 1e-3f);
        uniforms.orb[2] = config.rect.width * 0.5f / unit;
        uniforms.orb[3] = config.rect.height * 0.5f / unit;
        uniforms.params[0] = config.time;
        uniforms.params[1] = config.busy;
        uniforms.params[2] = config.activity;
        uniforms.params[3] = config.level;
        StoreColor(uniforms.colorA, config.listeningColor);
        StoreColor(uniforms.colorB, config.busyColor);
        StoreColor(uniforms.colorC, config.listeningAccent);
        StoreColor(uniforms.colorD, config.busyAccent);

        JST_CHECK(uniformBuffer->update());
        return Result::SUCCESS;
    }

    Result present() override {
        return Result::SUCCESS;
    }
};

OrbView::OrbView() {
    impl = std::make_unique<Impl>();
}

OrbView::~OrbView() = default;

bool OrbView::update(Config config) {
    impl->config = std::move(config);
    invalidate(Dirty::Paint);
    return true;
}

Result OrbView::build(Context& ctx) {
    ctx.drawables->push_back(impl.get());
    return impl->attach(&ctx, *ctx.surface);
}

Result OrbView::paint() {
    JST_CHECK(impl->upload());
    return Result::SUCCESS;
}

}  // namespace Jetstream::Sakura::Retained
