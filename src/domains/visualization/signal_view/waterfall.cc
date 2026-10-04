#include "waterfall.hh"

#include <algorithm>

#include "jetstream/constants.hh"
#include "resources/shaders/signal_view_shaders.hh"

namespace Jetstream::Modules {

void SignalViewWaterfall::configure(const Config& nextConfig) {
    config = nextConfig;
}

Result SignalViewWaterfall::create(const std::shared_ptr<Render::Window>& window,
                                   Tensor& bins,
                                   const std::string& colormap) {
    const U64 width = config.width;
    const U64 height = config.height;
    uniforms.filtered = width <= 65535 * 64 && bins.size() <= 16 * 1024 * 1024;
    filterKernel.reset();
    filteredBuffer.reset();
    filterUniformBuffer.reset();
    filterStateBuffer.reset();

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
        const U64 stride = width + 16;
        upload.assign(height * stride, 0.0f);
        for (U64 row = 0; row < height; ++row) {
            std::copy_n(bins.data<F32>() + row * width, width, upload.data() + row * stride);
        }
        cfg.buffer = upload.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = upload.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(binsBuffer, cfg));
    }

    if (uniforms.filtered) {
        filterUniforms = {static_cast<U32>(width), static_cast<U32>(height), 0, 0};
        filterState.assign(width, {});
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &filterUniforms;
            cfg.elementByteSize = sizeof(filterUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(filterUniformBuffer, cfg));
        }
        {
            Render::Buffer::Config cfg;
            cfg.buffer = filterState.data();
            cfg.elementByteSize = sizeof(filterState[0]);
            cfg.size = filterState.size();
            cfg.target = Render::Buffer::Target::STORAGE;
            JST_CHECK(window->build(filterStateBuffer, cfg));
        }
        {
            Render::Buffer::Config cfg;
            cfg.elementByteSize = sizeof(F32);
            cfg.size = bins.size();
            cfg.target = Render::Buffer::Target::STORAGE;
            JST_CHECK(window->build(filteredBuffer, cfg));
        }
        {
            Render::Kernel::Config cfg;
            cfg.gridSize = {width, 1, 1};
            cfg.workgroupSize = 64;
            cfg.kernels = KernelsPackage["waterfall_filter"];
            cfg.buffers = {
                {filterUniformBuffer, Render::Kernel::AccessMode::READ},
                {binsBuffer, Render::Kernel::AccessMode::READ},
                {filteredBuffer, Render::Kernel::AccessMode::WRITE},
                {filterStateBuffer, Render::Kernel::AccessMode::READ |
                                    Render::Kernel::AccessMode::WRITE},
            };
            JST_CHECK(window->build(filterKernel, cfg));
        }
    }

    JST_CHECK(lut.create(window, colormap));

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
        cfg.textures = {lut.texture()};
        cfg.buffers = {
            {uniformBuffer, Render::Program::Target::VERTEX |
                            Render::Program::Target::FRAGMENT},
            {binsBuffer, Render::Program::Target::FRAGMENT},
            {uniforms.filtered ? filteredBuffer : binsBuffer, Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(program, cfg));
    }

    return Result::SUCCESS;
}

void SignalViewWaterfall::attach(Render::Surface::Config& surface) const {
    if (filterKernel) {
        surface.kernels.push_back(filterKernel);
    }
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

Result SignalViewWaterfall::update(const WaterfallFrame& frame) {
    const U64 rows = frame.dirty.firstRowCount + frame.dirty.secondRowCount;
    if (filterKernel && rows > 0) {
        filterUniforms.writeIndex = static_cast<U32>(frame.writeIndex);
        filterUniforms.version += static_cast<U32>(rows);
        JST_CHECK(filterUniformBuffer->update());
        filterKernel->update();
    }
    const U64 stride = config.width + 16;
    for (U64 i = 0; i < rows; ++i) {
        const U64 row = (frame.dirty.startRow + i) % config.height;
        std::copy_n(frame.bins + row * config.width, config.width, upload.data() + row * stride);
    }
    if (frame.dirty.firstRowCount > 0) {
        JST_CHECK(binsBuffer->update(frame.dirty.startRow * stride,
                                     frame.dirty.firstRowCount * stride));
    }
    if (frame.dirty.secondRowCount > 0) {
        JST_CHECK(binsBuffer->update(0, frame.dirty.secondRowCount * stride));
    }
    uniforms.index = frame.writeIndex / static_cast<F32>(config.height);
    return Result::SUCCESS;
}

Result SignalViewWaterfall::present(const SurfaceInteractionState& interaction,
                                    const std::string& colormap) {
    JST_CHECK(lut.update(colormap));
    uniforms.width = static_cast<int>(config.width);
    uniforms.height = static_cast<int>(config.height);
    uniforms.offset = interaction.offset + 0.5f * (1.0f - 1.0f / interaction.zoom);
    uniforms.zoom = interaction.zoom;
    return uniformBuffer->update();
}

}  // namespace Jetstream::Modules
