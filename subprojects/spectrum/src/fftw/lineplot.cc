#include "spectrum/fftw/lineplot.hpp"
#include <random>

namespace Spectrum {

float get_random() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-0.5, 0.5); // rage 0 - 1
    return dis(e);
}

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

    for (float i = -1.0f; i < +1.0f; i += 1.0f/cfg.width) {
        l.push_back(i);
        l.push_back(get_random());
        l.push_back(+0.0f);
    }

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
            .usage = Render::Vertex::Buffer::Static,
        },
    };
    lineVertexCfg.mode = Render::Vertex::Mode::Lines;
    lineVertex = program->bind(lineVertexCfg);

    return Result::SUCCESS;
}

Result FFTW::LinePlot::destroy() {

    return Result::SUCCESS;
}

Result FFTW::LinePlot::draw() {
    for (int i = 0; i < l.size(); i += 3) {
        l.at(i+1) = get_random();
    }
    lineVertex->update();

    return Result::SUCCESS;
}

uint FFTW::LinePlot::raw() {
    return texture->raw();
}

} // namespace Spectrum

