#include "jetstream/histogram/cpu.hpp"
#include <algorithm>

namespace Jetstream::Histogram {

CPU::CPU(Config& c) : Generic(c) {
    auto render = cfg.render;

    buf.resize(in.buf.size() * 256);
    buf2.resize(in.buf.size() * 256);
    buf3.resize(in.buf.size() * 256);

    vertexCfg.buffers = Render::Extras::FillScreenVertices;
    vertexCfg.indices = Render::Extras::FillScreenIndices;
    vertex = render->create(vertexCfg);

    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Triangles;
    drawVertex = render->create(drawVertexCfg);

    binTextureCfg.height = 256;
    binTextureCfg.width = in.buf.size();
    binTextureCfg.buffer = (uint8_t*)buf3.data();
    binTextureCfg.key = "BinTexture";
    binTextureCfg.pfmt = Render::PixelFormat::RED;
    binTextureCfg.ptype = Render::PixelType::F32;
    binTextureCfg.dfmt = Render::DataFormat::F32;
    binTexture = render->create(binTextureCfg);

    lutTextureCfg.height = 1;
    lutTextureCfg.width = 256;
    lutTextureCfg.buffer = (uint8_t*)turbo_srgb_bytes;
    lutTextureCfg.key = "LutTexture";
    lutTexture = render->create(lutTextureCfg);

    programCfg.vertexSource = &vertexSource;
    programCfg.fragmentSource = &fragmentSource;
    programCfg.draws = {drawVertex};
    programCfg.textures = {binTexture, lutTexture};
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
    std::copy(in.buf.begin(), in.buf.end(), buf.begin()+(inc * in.buf.size()));

    for (int i = 0; i < in.buf.size(); i++) {
        buf2[(uint8_t)(in.buf[i] * 255.0) * in.buf.size() + i] += 1.0;
    }

    for (int i = 0; i < in.buf.size(); i++) {
        auto a = (uint8_t)(buf[i + ((((inc+1)% 256)*in.buf.size()))] * 255.0) * in.buf.size() + i;
        if (buf2[a] > 0.0) {
            buf2[a] -= 1.0;
        }
    }

    auto max = *std::max_element(buf2.begin(), buf2.end());
    auto min = *std::min_element(buf2.begin(), buf2.end());

    for (int i = 0; i < buf3.size(); i++) {
        buf3[i] = (buf2[i] - min)/(max - min);
    }

    inc += 1;
    if (inc >= 256) {
        inc = 0;
    }

    return Result::SUCCESS;
}

Result CPU::underlyingPresent() {
    binTexture->fill();
    program->setUniform("Index", std::vector<float>{inc/(float)cfg.height});
    vertex->update();
    return Result::SUCCESS;
}

} // namespace Jetstream::Histogram
