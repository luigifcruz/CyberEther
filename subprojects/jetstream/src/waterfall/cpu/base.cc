#include "jetstream/waterfall/cpu.hpp"

namespace Jetstream::Waterfall {

CPU::CPU(Config& c) : Generic(c) {
    auto render = cfg.render;

    buf.resize(in.buf.size() * cfg.height);

    vertexCfg.buffers = Render::Extras::FillScreenVertices;
    vertexCfg.indices = Render::Extras::FillScreenIndices;
    vertex = render->create(vertexCfg);

    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Triangles;
    drawVertex = render->create(drawVertexCfg);

    binTextureCfg.height = cfg.height;
    binTextureCfg.width = in.buf.size();
    binTextureCfg.buffer = (uint8_t*)buf.data();
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

    inc += 1;
    if (inc >= cfg.height) {
        inc = 0;
    }

    return Result::SUCCESS;
}

Result CPU::underlyingPresent() {
    // TODO: hot garbage, fix
    int start = last;
    int blocks = (inc - last);

    if (blocks < 0) {
        blocks = cfg.height - last;
        binTexture->fill(start, 0, in.buf.size(), blocks);
        start = 0;
        blocks = inc;
    }

    binTexture->fill(start, 0, in.buf.size(), blocks);
    last = inc;
    program->setUniform("Index", std::vector<float>{inc/(float)cfg.height});
    vertex->update();
    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
