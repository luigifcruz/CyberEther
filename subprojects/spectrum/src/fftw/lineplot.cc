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

    Render::Vertex::Config gridVertexCfg = {
        .buffers = {
            {
                .data = a.data(),
                .size = a.size(),
                .stride = 3,
                .usage = Render::Vertex::Buffer::Dynamic,
            },
        },
        .indices = p,
        .mode = Render::Vertex::Mode::Lines,
    };
    auto vertex = state.render->create<Render::GLES>(gridVertexCfg);

    Render::Program::Config programCfg = {
        .vertexSource = &vertexSource,
        .fragmentSource = &fragmentSource,
        .vertices = {vertex},
    };
    auto program = state.render->create<Render::GLES>(programCfg);

    Render::Texture::Config textureCfg = {
        .width = cfg.width,
        .height = cfg.height,
    };
    texture = state.render->create<Render::GLES>(textureCfg);

    Render::Surface::Config surfaceCfg = {
        .width = &textureCfg.width,
        .height = &textureCfg.height,
        .texture = texture,
        .programs = {program},
    };
    auto surface = state.render->create<Render::GLES>(surfaceCfg);

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

