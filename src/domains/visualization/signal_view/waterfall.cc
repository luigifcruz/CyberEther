#include "waterfall.hh"

#include "jetstream/constants.hh"
#include "resources/shaders/signal_view_shaders.hh"

namespace Jetstream::Modules {

void SignalViewWaterfall::configure(const Config& nextConfig) {
    config = nextConfig;
}

Result SignalViewWaterfall::create(const std::shared_ptr<Render::Window>& window, Tensor& bins) {
    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenVertices;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 12;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenVerticesBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenTextureVertices;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 8;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenTextureVerticesBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenIndices;
        cfg.elementByteSize = sizeof(U32);
        cfg.size = 6;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(fillScreenIndicesBuffer, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {fillScreenVerticesBuffer, 3},
            {fillScreenTextureVerticesBuffer, 2},
        };
        cfg.indices = fillScreenIndicesBuffer;
        JST_CHECK(window->build(vertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = vertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(draw, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = bins.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = bins.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(binsBuffer, cfg));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = const_cast<U8*>(&TurboLutBytes[0][0]);
        JST_CHECK(window->build(lutTexture, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &uniforms;
        cfg.elementByteSize = sizeof(uniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(uniformBuffer, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["waterfall"];
        cfg.draws = {draw};
        cfg.textures = {lutTexture};
        cfg.buffers = {
            {uniformBuffer, Render::Program::Target::VERTEX |
                            Render::Program::Target::FRAGMENT},
            {binsBuffer, Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(program, cfg));
    }

    return Result::SUCCESS;
}

void SignalViewWaterfall::attach(Render::Surface::Config& surface) const {
    surface.programs.push_back(program);
}

void SignalViewWaterfall::layout(const Render::ScissorRect& scissor,
                                 const F32 panelScaleX,
                                 const F32 panelScaleY,
                                 const F32 panelOffsetY) {
    program->scissorRect(scissor);
    uniforms.panelScaleX = panelScaleX;
    uniforms.panelScaleY = panelScaleY;
    uniforms.panelOffsetY = panelOffsetY;
}

Result SignalViewWaterfall::upload(const WaterfallFrame& frame) {
    if (frame.dirty.firstRowCount > 0) {
        JST_CHECK(binsBuffer->update(frame.dirty.startRow * config.width,
                                     frame.dirty.firstRowCount * config.width));
    }
    if (frame.dirty.secondRowCount > 0) {
        JST_CHECK(binsBuffer->update(0, frame.dirty.secondRowCount * config.width));
    }
    uniforms.index = frame.writeIndex / static_cast<F32>(config.height);
    return Result::SUCCESS;
}

Result SignalViewWaterfall::present(const SurfaceInteractionState& interaction) {
    uniforms.width = static_cast<int>(config.width);
    uniforms.height = static_cast<int>(config.height);
    uniforms.offset = interaction.offset + 0.5f * (1.0f - 1.0f / interaction.zoom);
    uniforms.zoom = interaction.zoom;
    return uniformBuffer->update();
}

}  // namespace Jetstream::Modules
