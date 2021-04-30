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

    for (float i = -1.0f; i < +1.0f; i += 1.0f/((float)cfg.input->size()/2)) {
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

    programCfg.vertexSource = &vertexSource;
    programCfg.fragmentSource = &fragmentSource;
    programCfg.draws = {drawGridVertex, drawLineVertex};
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
    for (int i = 0; i < cfg.input->size(); i++) {
        int ix;

        if (i < cfg.input->size() / 2) {
            ix = (cfg.input->size() / 2) + i;
        } else {
            ix = i - (cfg.input->size() / 2);
        }

        l[(i*3)+1] = ((20 * log10(abs(cfg.input->at(ix)) / cfg.input->size())) / (200.0 / 2)) + 1;
    }
    return Result::SUCCESS;
}

Result CPU::underlyingPresent() {
    lineVertex->update();
    return Result::SUCCESS;
}

} // namespace Jetstream::Lineplot
