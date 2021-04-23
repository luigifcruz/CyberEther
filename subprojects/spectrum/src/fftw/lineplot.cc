#include "spectrum/fftw/lineplot.hpp"

namespace Spectrum {

Result FFTW::LinePlot::create() {
    static std::vector<float> a;
    static std::vector<uint> p;
    int b = 0;
    for (float i = -1.0f; i < +1.0f; i += 0.10f) {
        a.push_back(-1.0f);
        a.push_back(i);
        a.push_back(+0.0f);
        p.push_back(b++);
        a.push_back(+1.0f);
        a.push_back(i);
        a.push_back(+0.0f);
        p.push_back(b++);
        a.push_back(i);
        a.push_back(-1.0f);
        a.push_back(+0.0f);
        p.push_back(b++);
        a.push_back(i);
        a.push_back(+1.0f);
        a.push_back(+0.0f);
        p.push_back(b++);
    }

    vertexCfg.buffers = {
        {
            .data = a.data(),
            .size = a.size(),
            .stride = 3,
            .usage = Render::Vertex::Buffer::Static,
        },
    };
    vertexCfg.indices = p;
    vertexCfg.mode = Render::Vertex::Mode::Lines;
    vertex = inst.cfg.render->create(vertexCfg);

    programCfg.vertexSource = &vertexSource;
    programCfg.fragmentSource = &fragmentSource;
    programCfg.vertices = {vertex};
    program = inst.cfg.render->create(programCfg);

    textureCfg.width = cfg.width;
    textureCfg.height = cfg.height;
    texture = inst.cfg.render->create(textureCfg);

    surfaceCfg.width = cfg.width;
    surfaceCfg.height = cfg.height;
    surfaceCfg.texture = texture;
    surfaceCfg.programs = {program};
    surface = inst.cfg.render->create(surfaceCfg);

    return Result::SUCCESS;
}

Result FFTW::LinePlot::destroy() {

    return Result::SUCCESS;
}

Result FFTW::LinePlot::draw() {

    return Result::SUCCESS;
}

uint FFTW::LinePlot::raw() {
    return texture->raw();
}

} // namespace Spectrum

