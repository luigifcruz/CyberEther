#include "jetstream/modules/lineplot.hh"

namespace Jetstream {

template<Device D, typename T>
Lineplot<D, T>::Lineplot(const Config& config,
                         const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Lineplot module.");
}

template<Device D, typename T>
const Result Lineplot<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Lineplot compute core using CPU backend.");
    
    // Generate Grid coordinates.
    {
        grid = Vector<D, F32>({20 * 12});
        int j = 0;
        for (float i = -1.0f; i < +1.0f; i += 0.10f) {
            grid[j++] = -1.0f;
            grid[j++] = i;
            grid[j++] = +0.0f;
            grid[j++] = +1.0f;
            grid[j++] = i;
            grid[j++] = +0.0f;
            grid[j++] = i;
            grid[j++] = -1.0f;
            grid[j++] = +0.0f;
            grid[j++] = i;
            grid[j++] = +1.0f;
            grid[j++] = +0.0f;
        }
    }
    
    // Generate Plot coordinates.
    {
        plot = Vector<D, F32>({input.buffer.shape(1) * 3});
        int j = 0;
        for (float i = -1.0f; i < +1.0f; i += 2.0f/((float)input.buffer.shape(1))) {
            plot[j++] = i;
            plot[j++] = +0.0f;
            plot[j++] = +0.0f;
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Lineplot<D, T>::createPresent(Render::Window& window) {
    Render::Buffer::Config gridVerticesConf;
    gridVerticesConf.buffer = grid.data();
    gridVerticesConf.elementByteSize = sizeof(grid[0]);
    gridVerticesConf.size = grid.size();
    gridVerticesConf.target = Render::Buffer::Target::VERTEX;
    gridVerticesConf.enableZeroCopy = true;
    JST_CHECK(window.build(gridVerticesBuffer, gridVerticesConf));

    Render::Vertex::Config gridVertexCfg;
    gridVertexCfg.buffers = {
        {gridVerticesBuffer, 3},
    };
    JST_CHECK(window.build(gridVertex, gridVertexCfg));

    Render::Draw::Config drawGridVertexCfg;
    drawGridVertexCfg.buffer = gridVertex;
    drawGridVertexCfg.mode = Render::Draw::Mode::LINES;
    JST_CHECK(window.build(drawGridVertex, drawGridVertexCfg));

    Render::Buffer::Config lineVerticesConf;
    lineVerticesConf.buffer = plot.data();
    lineVerticesConf.elementByteSize = sizeof(plot[0]);
    lineVerticesConf.size = plot.size();
    lineVerticesConf.target = Render::Buffer::Target::VERTEX;
    lineVerticesConf.enableZeroCopy = true;
    JST_CHECK(window.build(lineVerticesBuffer, lineVerticesConf));

    Render::Vertex::Config lineVertexCfg;
    lineVertexCfg.buffers = {
        {lineVerticesBuffer, 3},
    };
    JST_CHECK(window.build(lineVertex, lineVertexCfg));

    Render::Draw::Config drawLineVertexCfg;
    drawLineVertexCfg.buffer = lineVertex;
    drawLineVertexCfg.mode = Render::Draw::Mode::LINE_STRIP;
    JST_CHECK(window.build(drawLineVertex, drawLineVertexCfg));

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
    lutTextureCfg.key = "LutTexture";
    JST_CHECK(window.build(lutTexture, lutTextureCfg));

    Render::Program::Config programCfg;
    programCfg.shaders = {
        {Device::Metal, {MetalShader}},
    };
    programCfg.draws = {drawGridVertex, drawLineVertex};
    programCfg.textures = {lutTexture};
    JST_CHECK(window.build(program, programCfg));

    Render::Texture::Config textureCfg;
    textureCfg.size = config.viewSize;
    JST_CHECK(window.build(texture, textureCfg));

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    JST_CHECK(window.build(surface, surfaceCfg));
    JST_CHECK(window.bind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
void Lineplot<D, T>::summary() const {
    JST_INFO("===== Lineplot Module Configuration");
    JST_INFO("Size: {}x{}", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
const Result Lineplot<D, T>::present(Render::Window& window) {
    lineVerticesBuffer->update();
    return Result::SUCCESS;
}

template<Device D, typename T>
const Render::Size2D<U64>& Lineplot<D, T>::viewSize(const Render::Size2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        this->config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Lineplot<D, T>::getTexture() {
    return *texture;
};

}  // namespace Jetstream
