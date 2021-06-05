#include "jetstream/lineplot/cpu.hpp"

namespace Jetstream::Lineplot {

CPU::CPU(Config& c) : Generic(c) {
    for (float i = -1.0f; i < +1.0f; i += 0.10f) {
        a.push_back(-1.0f);
        a.push_back(i);
        a.push_back(+0.0f);
        a.push_back(+1.0f);
        a.push_back(i);
        a.push_back(+0.0f);
        a.push_back(i);
        a.push_back(-1.0f);
        a.push_back(+0.0f);
        a.push_back(i);
        a.push_back(+1.0f);
        a.push_back(+0.0f);
    }

    for (float i = -1.0f; i < +1.0f; i += 2.0f/((float)in.buf.size())) {
        l.push_back(i);
        l.push_back(+0.0f);
        l.push_back(+0.0f);
    }

    auto render = cfg.render;

    gridVertexCfg.buffers = {
        {
            .data = a.data(),
            .size = a.size(),
            .stride = 3,
            .usage = Render::Vertex::Buffer::Static,
        },
    };
    gridVertex = render->create(gridVertexCfg);

    drawGridVertexCfg.buffer = gridVertex;
    drawGridVertexCfg.mode = Render::Draw::Lines;
    drawGridVertex = render->create(drawGridVertexCfg);

    lineVertexCfg.buffers = {
        {
            .data = l.data(),
            .size = l.size(),
            .stride = 3,
            .usage = Render::Vertex::Buffer::Dynamic,
        },
    };
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
}

CPU::~CPU() {
}

float abs(std::complex<float> n) {
    return n.real() * n.real() + n.imag() * n.imag();
}

Result CPU::underlyingCompute() {
    float min_x = 0.0;
    float max_x = 1.0;

    for (int i = 0; i < in.buf.size(); i++) {
        l[(i*3)+1] = -(2 * ((in.buf[i] - min_x)/(max_x - min_x)) - 1);
    }
    return Result::SUCCESS;
}

Result CPU::underlyingPresent() {
    if (textureCfg.width != cfg.width || textureCfg.height != cfg.height) {
        if (surface->resize(cfg.width, cfg.height) != Render::Result::SUCCESS) {
            cfg.width = textureCfg.width;
            cfg.height = textureCfg.height;
        }
    }

    lineVertex->update();
    return Result::SUCCESS;
}

} // namespace Jetstream::Lineplot
