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
        l.push_back(get_random());
        l.push_back(+0.0f);
    }
    l.push_back(+1.0);
    l.push_back(-1.0);
    l.push_back(+0.0);

    std::cout << l.size() << std::endl;

    surface = inst.cfg.render->bind(surfaceCfg);

    textureCfg.width = &cfg.width;
    textureCfg.height = &cfg.height;
    texture = surface->bind(textureCfg);

    programCfg.vertexSource = &vertexSource;
    programCfg.fragmentSource = &fragmentSource;
    program = surface->bind(programCfg);

    gridVertexCfg.buffers = {
        {
            .data = a.data(),
            .size = a.size(),
            .stride = 3,
            .usage = Render::Vertex::Buffer::Static,
        },
    };
    gridVertexCfg.mode = Render::Vertex::Mode::Lines;
    gridVertex = program->bind(gridVertexCfg);

    lineVertexCfg.buffers = {
        {
            .data = l.data(),
            .size = l.size(),
            .stride = 3,
            .usage = Render::Vertex::Buffer::Dynamic,
        },
    };
    lineVertexCfg.mode = Render::Vertex::Mode::LineLoop;
    lineVertex = program->bind(lineVertexCfg);

    return Result::SUCCESS;
}

Result FFTW::LinePlot::destroy() {

    return Result::SUCCESS;
}

Result FFTW::LinePlot::draw() {
    for (int i = 0; i < l.size(); i += 3) {
        l.at(i+1) = log10(inst.fft_out[i/3].real() * inst.fft_out[i/3].real() + inst.fft_out[i/3].imag() * inst.fft_out[i/3].imag()) / 4.0 + 0.5;
    }
    lineVertex->update();

    return Result::SUCCESS;
}

uint FFTW::LinePlot::raw() {
    return texture->raw();
}

} // namespace Spectrum

