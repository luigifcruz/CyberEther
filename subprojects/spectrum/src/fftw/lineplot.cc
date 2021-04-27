#include "spectrum/fftw/lineplot.hpp"

namespace Spectrum {

Result FFTW::LinePlot::create() {
    static std::vector<float> a;
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

    l.push_back(-1.0);
    l.push_back(-1.0);
    l.push_back(+0.0);
    for (float i = -1.0f; i < +1.0f; i += 1.0f/(inst.cfg.size/2)) {
        l.push_back(i);
        l.push_back(+0.0f);
        l.push_back(+0.0f);
    }
    l.push_back(+1.0);
    l.push_back(-1.0);
    l.push_back(+0.0);

    auto render = inst.cfg.render;

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
    drawLineVertexCfg.mode = Render::Draw::LineLoop;
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
    surface = render->create(surfaceCfg);

    return Result::SUCCESS;
}

Result FFTW::LinePlot::destroy() {
    return Result::SUCCESS;
}

float abs(std::complex<float> n) {
    return n.real() * n.real() + n.imag() * n.imag();
}

Result FFTW::LinePlot::draw() {
    for (int i = 0; i < l.size(); i += 3) {
        l.at(i+1) = ((20 * log10(abs(inst.fft_out[i/3]) / inst.cfg.size)) / (200.0 / 2)) + 1;
    }
    lineVertex->update();

    return Result::SUCCESS;
}

uint FFTW::LinePlot::raw() {
    return texture->raw();
}

} // namespace Spectrum

