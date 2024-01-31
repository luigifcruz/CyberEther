#include "jetstream/modules/lineplot.hh"
#include "shaders/lineplot_shaders.hh"
#include "jetstream/render/utils.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result Lineplot<D, T>::create() {
    JST_DEBUG("Initializing Lineplot module.");
    JST_INIT_IO();

    // Check parameters.

    if (input.buffer.rank() > 2) {
        JST_ERROR("Invalid input rank ({}). It should be `1` or `2`.", input.buffer.rank());
        return Result::ERROR;
    }

    // Calculate parameters.

    const U64 last_axis = input.buffer.rank() - 1;
    numberOfElements = input.buffer.shape()[last_axis];
    numberOfBatches = (input.buffer.rank() == 2) ? input.buffer.shape()[0] : 1;

    // Allocate internal buffers.

    {
        // Generate Grid coordinates.

        const U64 num_cols = config.numberOfVerticalLines;
        const U64 num_rows = config.numberOfHorizontalLines;

        auto tmp = Tensor<Device::CPU, F32>({num_cols + num_rows, 2, 3});

        const F32 x_step  = +2.0f / (num_cols - 1);
        const F32 y_step  = +2.0f / (num_rows - 1);
        const F32 x_start = -1.0f;
        const F32 y_start = -1.0f;
        const F32 x_end   = +1.0f;
        const F32 y_end   = +1.0f;

        for (U64 row = 0; row < num_rows; row++) {
            const F32 y = y_start + row * y_step;

            tmp[{row, 0, 0}] = x_start;
            tmp[{row, 0, 1}] = y;

            tmp[{row, 1, 0}] = x_end;
            tmp[{row, 1, 1}] = y;
        }

        for (U64 col = 0; col < num_cols; col++) {
            const F32 x = x_start + col * x_step;

            tmp[{col + num_rows, 0, 0}] = x;
            tmp[{col + num_rows, 0, 1}] = y_start;

            tmp[{col + num_rows, 1, 0}] = x;
            tmp[{col + num_rows, 1, 1}] = y_end;
        }

        grid = Tensor<D, T>(tmp.shape());
        JST_CHECK(Memory::Copy(grid, MapOn<D>(tmp)));
    }

    {
        // Generate Plot coordinates.

        auto tmp = Tensor<Device::CPU, F32>({numberOfElements, 3});

        for (U64 j = 0; j < numberOfElements; j++) {
            tmp[{j, 0}] = j * 2.0f / (numberOfElements - 1) - 1.0f;
        }

        plot = Tensor<D, T>(tmp.shape());
        JST_CHECK(Memory::Copy(plot, MapOn<D>(tmp)));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
void Lineplot<D, T>::info() const {
    JST_INFO("  Size: [{}, {}]", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
Result Lineplot<D, T>::createPresent() {
    auto [gridBuffer, gridEnableZeroCopy] = ConvertToOptimalStorage(window, grid);

    Render::Buffer::Config gridVerticesConf;
    gridVerticesConf.buffer = gridBuffer;
    gridVerticesConf.elementByteSize = sizeof(F32);
    gridVerticesConf.size = grid.size();
    gridVerticesConf.target = Render::Buffer::Target::VERTEX;
    gridVerticesConf.enableZeroCopy = gridEnableZeroCopy;
    JST_CHECK(window->build(gridVerticesBuffer, gridVerticesConf));

    Render::Vertex::Config gridVertexCfg;
    gridVertexCfg.buffers = {
        {gridVerticesBuffer, 3},
    };
    JST_CHECK(window->build(gridVertex, gridVertexCfg));

    Render::Draw::Config drawGridVertexCfg;
    drawGridVertexCfg.buffer = gridVertex;
    drawGridVertexCfg.mode = Render::Draw::Mode::LINES;
    JST_CHECK(window->build(drawGridVertex, drawGridVertexCfg));

    auto [plotBuffer, plotEnableZeroCopy] = ConvertToOptimalStorage(window, plot);

    Render::Buffer::Config lineVerticesConf;
    lineVerticesConf.buffer = plotBuffer;
    lineVerticesConf.elementByteSize = sizeof(F32);
    lineVerticesConf.size = plot.size();
    lineVerticesConf.target = Render::Buffer::Target::VERTEX;
    lineVerticesConf.enableZeroCopy = plotEnableZeroCopy;
    JST_CHECK(window->build(lineVerticesBuffer, lineVerticesConf));

    Render::Vertex::Config lineVertexCfg;
    lineVertexCfg.buffers = {
        {lineVerticesBuffer, 3},
    };
    JST_CHECK(window->build(lineVertex, lineVertexCfg));

    Render::Draw::Config drawLineVertexCfg;
    drawLineVertexCfg.buffer = lineVertex;
    drawLineVertexCfg.mode = Render::Draw::Mode::LINE_STRIP;
    JST_CHECK(window->build(drawLineVertex, drawLineVertexCfg));

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
    JST_CHECK(window->build(lutTexture, lutTextureCfg));

    Render::Program::Config gridProgramCfg;
    gridProgramCfg.shaders = ShadersPackage["grid"];
    gridProgramCfg.draw = drawGridVertex;
    JST_CHECK(window->build(gridProgram, gridProgramCfg));

    Render::Program::Config signalProgramCfg;
    signalProgramCfg.shaders = ShadersPackage["signal"];
    signalProgramCfg.draw = drawLineVertex;
    signalProgramCfg.textures = {lutTexture};
    JST_CHECK(window->build(signalProgram, signalProgramCfg));

    Render::Texture::Config textureCfg;
    textureCfg.size = config.viewSize;
    JST_CHECK(window->build(texture, textureCfg));

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {gridProgram, signalProgram};
    JST_CHECK(window->build(surface, surfaceCfg));
    JST_CHECK(window->bind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::present() {
    lineVerticesBuffer->update();
    return Result::SUCCESS;
}

template<Device D, typename T>
const Size2D<U64>& Lineplot<D, T>::viewSize(const Size2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Lineplot<D, T>::getTexture() {
    return *texture;
};

}  // namespace Jetstream
