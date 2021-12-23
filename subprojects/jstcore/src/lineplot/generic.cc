#include "jstcore/lineplot/shaders.hpp"
#include "jstcore/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

Generic::Generic(const Config& config, const Input& input) : config(config), input(input) {
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

    for (float i = -1.0f; i < +1.0f; i += 2.0f/((float)input.in.buf.size())) {
        plot.push_back(i);
        plot.push_back(+0.0f);
        plot.push_back(+0.0f);
    }
}

Result Generic::initRender(float* ptr, bool cudaInterop) {
    Render::Vertex::Config gridVertexCfg;
    Render::Vertex::Buffer gridVbo;
    gridVbo.data = grid.data();
    gridVbo.size = grid.size();
    gridVbo.stride = 3;
    gridVbo.usage = Render::Vertex::Buffer::Static;
    gridVertexCfg.buffers = {gridVbo};
    gridVertex = Render::Create(gridVertexCfg);

    Render::Draw::Config drawGridVertexCfg;
    drawGridVertexCfg.buffer = gridVertex;
    drawGridVertexCfg.mode = Render::Draw::Lines;
    drawGridVertex = Render::Create(drawGridVertexCfg);

    Render::Vertex::Config lineVertexCfg;
    Render::Vertex::Buffer plotVbo;
    plotVbo.size = plot.size();
    plotVbo.stride = 3;
    plotVbo.cudaInterop = cudaInterop;
    plotVbo.data = ptr;
    plotVbo.usage = Render::Vertex::Buffer::Dynamic;
    lineVertexCfg.buffers = {plotVbo};
    lineVertex = Render::Create(lineVertexCfg);

    Render::Draw::Config drawLineVertexCfg;
    drawLineVertexCfg.buffer = lineVertex;
    drawLineVertexCfg.mode = Render::Draw::LineStrip;
    drawLineVertex = Render::Create(drawLineVertexCfg);

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)turbo_srgb_bytes;
    lutTextureCfg.key = "LutTexture";
    lutTexture = Render::Create(lutTextureCfg);

    Render::Program::Config programCfg;
    programCfg.shaders = {
        {Render::Backend::Metal, {MetalShader}},
        {Render::Backend::GLES, {GlesVertexShader, GlesFragmentShader}},
    };
    programCfg.draws = {drawGridVertex, drawLineVertex};
    programCfg.textures = {lutTexture};
    program = Render::Create(programCfg);

    Render::Texture::Config textureCfg;
    textureCfg.size = config.size;
    texture = Render::Create(textureCfg);

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    surface = Render::CreateAndBind(surfaceCfg);

    return Result::SUCCESS;
}

Result Generic::compute() {
    return this->underlyingCompute();
}

Result Generic::present() {
    lineVertex->update();
    return Result::SUCCESS;
}

Size2D<int> Generic::size(const Size2D<int>& size) {
    if (surface->size(size) != this->size()) {
        config.size = surface->size();
    }
    return this->size();
}

std::weak_ptr<Render::Texture> Generic::tex() const {
    return texture;
};

} // namespace Jetstream::Lineplot
