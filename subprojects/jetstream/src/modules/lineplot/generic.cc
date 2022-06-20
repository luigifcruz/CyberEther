#include "jetstream/modules/lineplot.hh"

namespace Jetstream {

template<Device D, typename T>
const Result Lineplot<D, T>::initializeRender() {
    Render::Buffer::Config gridVerticesConf;
    gridVerticesConf.buffer = grid.data();
    gridVerticesConf.elementByteSize = sizeof(grid[0]);
    gridVerticesConf.size = grid.size();
    gridVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK_THROW(Render::Create(gridVerticesBuffer, gridVerticesConf));

    Render::Vertex::Config gridVertexCfg;
    gridVertexCfg.buffers = {
        {gridVerticesBuffer, 3},
    };
    JST_CHECK_THROW(Render::Create(gridVertex, gridVertexCfg));

    Render::Draw::Config drawGridVertexCfg;
    drawGridVertexCfg.buffer = gridVertex;
    drawGridVertexCfg.mode = Render::Draw::Mode::LINES;
    JST_CHECK_THROW(Render::Create(drawGridVertex, drawGridVertexCfg));

    Render::Buffer::Config lineVerticesConf;
    lineVerticesConf.buffer = plot.data();
    lineVerticesConf.elementByteSize = sizeof(plot[0]);
    lineVerticesConf.size = plot.size();
    lineVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK_THROW(Render::Create(lineVerticesBuffer, lineVerticesConf));

    Render::Vertex::Config lineVertexCfg;
    lineVertexCfg.buffers = {
        {lineVerticesBuffer, 3},
    };
    JST_CHECK_THROW(Render::Create(lineVertex, lineVertexCfg));

    Render::Draw::Config drawLineVertexCfg;
    drawLineVertexCfg.buffer = lineVertex;
    drawLineVertexCfg.mode = Render::Draw::Mode::LINE_STRIP;
    JST_CHECK_THROW(Render::Create(drawLineVertex, drawLineVertexCfg));

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
    lutTextureCfg.key = "LutTexture";
    JST_CHECK_THROW(Render::Create(lutTexture, lutTextureCfg));

    Render::Program::Config programCfg;
    programCfg.shaders = {
        {Device::Metal, {MetalShader}},
    };
    programCfg.draws = {drawGridVertex, drawLineVertex};
    programCfg.textures = {lutTexture};
    JST_CHECK_THROW(Render::Create(program, programCfg));

    Render::Texture::Config textureCfg;
    textureCfg.size = config.viewSize;
    JST_CHECK_THROW(Render::Create(texture, textureCfg));

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    JST_CHECK_THROW(Render::Create(surface, surfaceCfg));

    return Result::SUCCESS;
}

template<Device D, typename T>
Lineplot<D, T>::Lineplot(const Config& config, const Input& input) 
    : config(config), input(input) {
    JST_DEBUG("Initializing Lineplot module.");

    // Generate Grid and Plot coordinates.
    for (float i = -1.0f; i < +1.0f; i += 0.10f) {
        grid.push_back(-1.0f);
        grid.push_back(i);
        grid.push_back(+0.0f);
        grid.push_back(+1.0f);
        grid.push_back(i);
        grid.push_back(+0.0f);
        grid.push_back(i);
        grid.push_back(-1.0f);
        grid.push_back(+0.0f);
        grid.push_back(i);
        grid.push_back(+1.0f);
        grid.push_back(+0.0f);
    }

    for (float i = -1.0f; i < +1.0f; i += 2.0f/((float)getBufferSize())) {
        plot.push_back(i);
        plot.push_back(+0.0f);
        plot.push_back(+0.0f);
    }

    JST_CHECK_THROW(initializeRender());
    
    JST_INFO("===== Lineplot Module Configuration");
    JST_INFO("Size: {}x{}", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
const Result Lineplot<D, T>::present(const RuntimeMetadata& meta) {
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
