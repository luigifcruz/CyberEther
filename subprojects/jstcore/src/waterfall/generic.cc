#include "jstcore/waterfall/generic.hpp"

namespace Jetstream::Waterfall {

Generic::Generic(const Config & config, const Input & input) : config(config), input(input) {}

Result Generic::initRender(uint8_t* ptr, bool cudaInterop) {
    if (!config.render) {
        std::cerr << "[JETSTREAM:WATERFALL] Invalid Render pointer" << std::endl;
        return Result::ERROR;
    }

    auto render = config.render;

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = Render::Extras::FillScreenVertices();
    vertexCfg.indices = Render::Extras::FillScreenIndices();
    vertex = render->create(vertexCfg);

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Triangles;
    drawVertex = render->create(drawVertexCfg);

    Render::Texture::Config binTextureCfg;
    binTextureCfg.size = {static_cast<int>(input.in.buf.size()), ymax};
    binTextureCfg.buffer = ptr;
    binTextureCfg.cudaInterop = cudaInterop;
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
    textureCfg.size = config.size;
    texture = render->create(textureCfg);

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    surface = render->createAndBind(surfaceCfg);

    return Result::SUCCESS;
}

Result Generic::compute() {
    auto res = this->underlyingCompute();
    inc = (inc + 1) % ymax;
    return res;
}

Result Generic::present() {
    int start = last;
    int blocks = (inc - last);

    // TODO: Fix this horrible thing.
    if (blocks < 0) {
        blocks = ymax - last;
        binTexture->fill(start, 0, input.in.buf.size(), blocks);
        start = 0;
        blocks = inc;
    }

    binTexture->fill(start, 0, input.in.buf.size(), blocks);
    last = inc;

    program->setUniform("Index", std::vector<float>{inc/(float)ymax});
    program->setUniform("Interpolate", std::vector<int>{(int)config.interpolate});
    vertex->update();

    return Result::SUCCESS;
}

bool Generic::interpolate(bool val) {
    config.interpolate = val;
    return this->interpolate();
}

Size2D<int> Generic::size(const Size2D<int> & size) {
    if (surface->size(size) != this->size()) {
        config.size = surface->size();
    }
    return this->size();
}

std::weak_ptr<Render::Texture> Generic::tex() const {
    return texture;
};

} // namespace Jetstream::Waterfall
