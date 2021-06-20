#include "jetstream/lineplot/generic.hpp"

namespace Jetstream {
namespace Lineplot {

Generic::Generic(const Config & c) : Module(cfg.policy), cfg(c), in(cfg.input0) {
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

    for (float i = -1.0f; i < +1.0f; i += 2.0f/((float)in.buf.size())) {
        plot.push_back(i);
        plot.push_back(+0.0f);
        plot.push_back(+0.0f);
    }
}

Result Generic::_initRender(float* ptr, bool cudaInterop) {
    if (!cfg.render) {
        std::cerr << "[JETSTREAM:LINEPLOT] Invalid Render pointer" << std::endl;
        CHECK(Result::ERROR);
    }

    auto render = cfg.render;

    Render::Vertex::Config gridVertexCfg;
    Render::Vertex::Buffer gridVbo;
    gridVbo.data = grid.data();
    gridVbo.size = grid.size();
    gridVbo.stride = 3;
    gridVbo.usage = Render::Vertex::Buffer::Static;
    gridVertexCfg.buffers = {gridVbo};
    gridVertex = render->create(gridVertexCfg);

    Render::Draw::Config drawGridVertexCfg;
    drawGridVertexCfg.buffer = gridVertex;
    drawGridVertexCfg.mode = Render::Draw::Lines;
    drawGridVertex = render->create(drawGridVertexCfg);

    Render::Vertex::Config lineVertexCfg;
    Render::Vertex::Buffer plotVbo;
    plotVbo.size = plot.size();
    plotVbo.stride = 3;
    plotVbo.cudaInterop = cudaInterop;
    plotVbo.data = ptr;
    plotVbo.usage = Render::Vertex::Buffer::Dynamic;
    lineVertexCfg.buffers = {plotVbo};
    lineVertex = render->create(lineVertexCfg);

    Render::Draw::Config drawLineVertexCfg;
    drawLineVertexCfg.buffer = lineVertex;
    drawLineVertexCfg.mode = Render::Draw::LineStrip;
    drawLineVertex = render->create(drawLineVertexCfg);

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)turbo_srgb_bytes;
    lutTextureCfg.key = "LutTexture";
    lutTexture = render->create(lutTextureCfg);

    Render::Program::Config programCfg;
    programCfg.vertexSource = &vertexSource;
    programCfg.fragmentSource = &fragmentSource;
    programCfg.draws = {drawGridVertex, drawLineVertex};
    programCfg.textures = {lutTexture};
    program = render->create(programCfg);

    Render::Texture::Config textureCfg;
    textureCfg.size = cfg.size;
    texture = render->create(textureCfg);

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    surface = render->createAndBind(surfaceCfg);

    return Result::SUCCESS;
}

Result Generic::underlyingCompute() {
    DEBUG_PUSH("compute_lineplot");

    auto res = this->_compute();

    DEBUG_POP();
    return res;
}

Result Generic::underlyingPresent() {
    DEBUG_PUSH("present_lineplot");

    auto res = this->_present();

    DEBUG_POP();
    return res;
}

Size2D<int> Generic::size(const Size2D<int> & size) {
    if (surface->size(size) != cfg.size) {
        cfg.size = surface->size();
    }

    return this->size();
}

std::weak_ptr<Render::Texture> Generic::tex() const {
    return texture;
};

} // namespace Lineplot
} // namespace Jetstream