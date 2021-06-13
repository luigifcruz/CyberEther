#include "jetstream/waterfall/generic.hpp"

namespace Jetstream::Waterfall {

Generic::Generic(Config& c) : Module(c.policy), cfg(c), in(c.input0) {
}

Result Generic::_initRender() {
    auto render = cfg.render;

    vertexCfg.buffers = Render::Extras::FillScreenVertices;
    vertexCfg.indices = Render::Extras::FillScreenIndices;
    vertex = render->create(vertexCfg);

    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Triangles;
    drawVertex = render->create(drawVertexCfg);

    binTextureCfg.height = ymax;
    binTextureCfg.width = in.buf.size();
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

    return Result::SUCCESS;
}

Result Generic::underlyingCompute() {
    DEBUG_PUSH("compute_waterfall");
    auto res = this->_compute();

    inc = (inc + 1) % ymax;

    DEBUG_POP();
    return res;
}

Result Generic::underlyingPresent() {
    DEBUG_PUSH("present_waterfall");
    if (textureCfg.width != cfg.width || textureCfg.height != cfg.height) {
        if (surface->resize(cfg.width, cfg.height) != Render::Result::SUCCESS) {
            cfg.width = textureCfg.width;
            cfg.height = textureCfg.height;
        }
    }

    auto res = this->_present();

    program->setUniform("Index", std::vector<float>{inc/(float)ymax});
    program->setUniform("Interpolate", std::vector<int>{(int)cfg.interpolate});
    vertex->update();

    DEBUG_POP();
    return res;
}

} // namespace Jetstream::Waterfall
