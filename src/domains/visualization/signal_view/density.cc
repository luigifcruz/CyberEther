#include "density.hh"

#include "jetstream/constants.hh"
#include "resources/shaders/signal_view_shaders.hh"

namespace Jetstream::Modules {

void SignalViewDensity::configure(const Config& nextConfig) {
    config = nextConfig;
    columns = detail::PlanDensityColumns(config.width);

    accumulateUniforms = {};
    accumulateUniforms.width = static_cast<U32>(config.width);
    accumulateUniforms.columns = static_cast<U32>(columns.columns);
    accumulateUniforms.levels = static_cast<U32>(detail::kDensityLevels);
    accumulateUniforms.binsPerColumn = static_cast<U32>(columns.binsPerColumn);
    accumulateUniforms.history = static_cast<U32>(config.history);
    accumulateUniforms.epoch = 1;

    uniforms = {};
    uniforms.columns = static_cast<U32>(columns.columns);
    uniforms.levels = static_cast<U32>(detail::kDensityLevels);
    uniforms.binsPerColumn = static_cast<F32>(columns.binsPerColumn);
    uniforms.binSpan = static_cast<F32>(config.width - 1);

    persistence(config.persistence);
}

Result SignalViewDensity::create(const std::shared_ptr<Render::Window>& window,
                                 Tensor& history) {
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
        cfg.buffer = history.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = history.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(historyBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = columns.columns * detail::kDensityLevels;
        cfg.target = Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(densityBuffer, cfg));
    }

    {
        state.assign(columns.columns, {0, 0});
        Render::Buffer::Config cfg;
        cfg.buffer = state.data();
        cfg.elementByteSize = sizeof(state[0]);
        cfg.size = state.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(stateBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &accumulateUniforms;
        cfg.elementByteSize = sizeof(accumulateUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(accumulateUniformBuffer, cfg));
    }

    {
        Render::Kernel::Config cfg;
        cfg.gridSize = {columns.columns, 1, 1};
        cfg.workgroupSize = 64;
        cfg.kernels = KernelsPackage["density_accumulate"];
        cfg.buffers = {
            {accumulateUniformBuffer, Render::Kernel::AccessMode::READ},
            {historyBuffer, Render::Kernel::AccessMode::READ},
            {densityBuffer, Render::Kernel::AccessMode::READ |
                            Render::Kernel::AccessMode::WRITE},
            {stateBuffer, Render::Kernel::AccessMode::READ |
                          Render::Kernel::AccessMode::WRITE},
        };
        JST_CHECK(window->build(accumulateKernel, cfg));
    }

    JST_CHECK(lut.create(window, detail::kDensityColormap));

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
        cfg.shaders = ShadersPackage["density"];
        cfg.draws = {draw};
        cfg.textures = {lut.texture()};
        cfg.buffers = {
            {uniformBuffer, Render::Program::Target::VERTEX |
                            Render::Program::Target::FRAGMENT},
            {densityBuffer, Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(program, cfg));
    }

    return Result::SUCCESS;
}

void SignalViewDensity::attach(Render::Surface::Config& surface) const {
    surface.kernels.push_back(accumulateKernel);
    surface.programs.push_back(program);
}

void SignalViewDensity::layout(const Render::ScissorRect& scissor,
                               const F32 panelScaleX,
                               const F32 panelScaleY,
                               const F32 panelOffsetY) {
    program->scissorRect(scissor);
    uniforms.panelScaleX = panelScaleX;
    uniforms.panelScaleY = panelScaleY;
    uniforms.panelOffsetY = panelOffsetY;
    uniforms.plotWidth = static_cast<F32>(scissor.width);
}

void SignalViewDensity::persistence(const U64 halfLife) {
    config.persistence = halfLife;
    accumulateUniforms.decay = detail::DensityDecay(static_cast<F32>(halfLife));
    uniforms.intensityScale = 1.0f - accumulateUniforms.decay;
    accumulateDirty = true;
}

void SignalViewDensity::reset() {
    accumulateUniforms.origin = accumulateUniforms.version;
    accumulateUniforms.epoch += 1;
    accumulateDirty = true;
}

Result SignalViewDensity::update(const WaterfallFrame& frame) {
    const U64 rows = frame.dirty.firstRowCount + frame.dirty.secondRowCount;
    if (rows == 0 && frame.elapsedRows == 0) {
        return Result::SUCCESS;
    }
    if (frame.dirty.firstRowCount > 0) {
        JST_CHECK(historyBuffer->update(frame.dirty.startRow * config.width,
                                        frame.dirty.firstRowCount * config.width));
    }
    if (frame.dirty.secondRowCount > 0) {
        JST_CHECK(historyBuffer->update(0, frame.dirty.secondRowCount * config.width));
    }
    accumulateUniforms.writeIndex = static_cast<U32>(frame.writeIndex);
    accumulateUniforms.version += static_cast<U32>(frame.elapsedRows);
    accumulateDirty = true;
    return Result::SUCCESS;
}

Result SignalViewDensity::present(const SurfaceInteractionState& interaction) {
    uniforms.offset = interaction.offset + 0.5f * (1.0f - 1.0f / interaction.zoom);
    uniforms.zoom = interaction.zoom;
    JST_CHECK(uniformBuffer->update());
    if (accumulateDirty) {
        JST_CHECK(accumulateUniformBuffer->update());
        accumulateKernel->update();
        accumulateDirty = false;
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
