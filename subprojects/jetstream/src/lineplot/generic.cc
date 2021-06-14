#include "jetstream/lineplot/generic.hpp"

namespace Jetstream::Lineplot {

Generic::Generic(Config& c) : Module(c.policy), cfg(c), in(c.input0) {
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

Result Generic::_initRender() {
    if (!cfg.render) {
        std::cerr << "[JETSTREAM:LINEPLOT] Invalid Render pointer" << std::endl;
        JETSTREAM_ASSERT_SUCCESS(Result::ERROR);
    }

    auto render = cfg.render;

    gridVbo.data = grid.data();
    gridVbo.size = grid.size();
    gridVbo.stride = 3;
    gridVbo.usage = Render::Vertex::Buffer::Static;
    gridVertexCfg.buffers = {gridVbo};
    gridVertex = render->create(gridVertexCfg);

    drawGridVertexCfg.buffer = gridVertex;
    drawGridVertexCfg.mode = Render::Draw::Lines;
    drawGridVertex = render->create(drawGridVertexCfg);

    plotVbo.size = plot.size();
    plotVbo.stride = 3;
    plotVbo.usage = Render::Vertex::Buffer::Dynamic;
    lineVertexCfg.buffers = {plotVbo};
    lineVertex = render->create(lineVertexCfg);

    drawLineVertexCfg.buffer = lineVertex;
    drawLineVertexCfg.mode = Render::Draw::LineStrip;
    drawLineVertex = render->create(drawLineVertexCfg);

    lutTextureCfg.height = 1;
    lutTextureCfg.width = 256;
    lutTextureCfg.buffer = (uint8_t*)turbo_srgb_bytes;
    lutTextureCfg.key = "LutTexture";
    lutTexture = render->create(lutTextureCfg);

    programCfg.vertexSource = &vertexSource;
    programCfg.fragmentSource = &fragmentSource;
    programCfg.draws = {drawGridVertex, drawLineVertex};
    programCfg.textures = {lutTexture};
    program = render->create(programCfg);

    textureCfg.width = cfg.width;
    textureCfg.height = cfg.height;
    texture = render->create(textureCfg);

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
    if (textureCfg.width != cfg.width || textureCfg.height != cfg.height) {
        if (surface->resize(cfg.width, cfg.height) != Render::Result::SUCCESS) {
            cfg.width = textureCfg.width;
            cfg.height = textureCfg.height;
        }
    }

    auto res = this->_present();
    DEBUG_POP();
    return res;
}

} // namespace Jetstream::Waterfall

