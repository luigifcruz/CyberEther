#include "jetstream/histogram/cpu.hpp"

namespace Jetstream {

Histogram::CPU::CPU(const Config & c) : Histogram(c) {
    auto render = cfg.render;

    buf.resize(in.buf.size() * 256);
    buf2.resize(in.buf.size() * 256);

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = Render::Extras::FillScreenVertices();
    vertexCfg.indices = Render::Extras::FillScreenIndices();
    vertex = render->create(vertexCfg);

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Triangles;
    drawVertex = render->create(drawVertexCfg);

    Render::Texture::Config binTextureCfg;
    binTextureCfg.size = {static_cast<int>(in.buf.size()), 256};
    binTextureCfg.buffer = (uint8_t*)buf2.data();
    binTextureCfg.key = "BinTexture";
    binTextureCfg.pfmt = Render::PixelFormat::RED;
    binTextureCfg.ptype = Render::PixelType::F32;
    binTextureCfg.dfmt = Render::DataFormat::F32;
    binTexture = render->create(binTextureCfg);

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)turbo_srgb_bytes;
    lutTextureCfg.key = "LutTexture";
    lutTexture = render->create(lutTextureCfg);

    Render::Program::Config programCfg;
    programCfg.vertexSource = &vertexSource;
    programCfg.fragmentSource = &fragmentSource;
    programCfg.draws = {drawVertex};
    programCfg.textures = {binTexture, lutTexture};
    program = render->create(programCfg);

    Render::Texture::Config textureCfg;
    textureCfg.size = cfg.size;
    texture = render->create(textureCfg);

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    surface = render->createAndBind(surfaceCfg);
}

Histogram::CPU::~CPU() {
}

float abs(std::complex<float> n) {
    return n.real() * n.real() + n.imag() * n.imag();
}

Result Histogram::CPU::underlyingCompute() {
    std::copy(in.buf.begin(), in.buf.end(), buf.begin()+(inc * in.buf.size()));

    for (int i = 0; i < in.buf.size(); i++) {
        buf2[(uint8_t)(in.buf[i] * 255.0) * in.buf.size() + i] += 1.0 / 50.0;
    }

    for (int i = 0; i < in.buf.size(); i++) {
        buf2[(uint8_t)(buf[i + ((((inc+1)% 256)*in.buf.size()))] * 255.0) * in.buf.size() + i] -= 1.0 / 50.0;
    }

    inc += 1;
    if (inc >= 256) {
        inc = 0;
    }

    return Result::SUCCESS;
}

Result Histogram::CPU::underlyingPresent() {
    binTexture->fill();
    program->setUniform("Index", std::vector<float>{inc/(float)cfg.size.height});
    vertex->update();
    return Result::SUCCESS;
}

} // namespace Jetstream
