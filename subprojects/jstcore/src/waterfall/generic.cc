#include "jstcore/waterfall/shaders.hpp"
#include "jstcore/waterfall/generic.hpp"

namespace Jetstream::Waterfall {

Generic::Generic(const Config& config, const Input& input) : config(config), input(input) {}

Result Generic::initRender(uint8_t* ptr, bool cudaInterop) {
    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = Render::Extras::FillScreenVertices();
    vertexCfg.indices = Render::Extras::FillScreenIndices();
    vertex = Render::Create(vertexCfg);

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Triangles;
    drawVertex = Render::Create(drawVertexCfg);

    Render::Texture::Config binTextureCfg;
    binTextureCfg.size = {static_cast<int>(input.in.buf.size()), ymax};
    binTextureCfg.buffer = ptr;
    binTextureCfg.cudaInterop = cudaInterop;
    binTextureCfg.key = "BinTexture";
    binTextureCfg.pfmt = Render::PixelFormat::RED;
    binTextureCfg.ptype = Render::PixelType::F32;
    binTextureCfg.dfmt = Render::DataFormat::F32;
    binTexture = Render::Create(binTextureCfg);

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)turbo_srgb_bytes;
    lutTextureCfg.key = "LutTexture";
    lutTexture = Render::Create(lutTextureCfg);

    Render::Program::Config programCfg;
    programCfg.shaders = {
        {Render::Backend::Metal, {MetalShader}},
        {Render::Backend::GLES, {GlesVertexShader, GlesFragmentShader}},
    };
    programCfg.draws = {drawVertex};
    programCfg.textures = {binTexture, lutTexture};
    programCfg.uniforms = {
        {"index", &indexUniform},
        {"interpolate", &interpolateUniform},
    };
    program = Render::Create(programCfg);

    Render::Texture::Config textureCfg;
    textureCfg.size = config.size;
    texture = Render::Create(textureCfg);

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    surface = Render::CreateAndBind(surfaceCfg);

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

        binTexture->fillRow(start, blocks);

        start = 0;
        blocks = inc;
    }

    binTexture->fillRow(start, blocks);
    last = inc;

    indexUniform[0] = inc / (float)ymax;
    interpolateUniform[0] = config.interpolate;

    return Result::SUCCESS;
}

bool Generic::interpolate(bool val) {
    config.interpolate = val;
    return this->interpolate();
}

Size2D<int> Generic::size(const Size2D<int>& size) {
    if (surface->size(size) != this->size()) {
        config.size = surface->size();
    }
    return this->size();
}

Render::Texture& Generic::tex() const {
    return *texture;
};

} // namespace Jetstream::Waterfall
