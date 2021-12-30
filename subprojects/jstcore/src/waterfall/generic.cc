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

    Render::Buffer::Config bufferCfg;
    bufferCfg.size = input.in.buf.size() * ymax * sizeof(float);
    bufferCfg.buffer = ptr;
    binTexture = Render::Create(bufferCfg);

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
    programCfg.textures = {lutTexture};
    programCfg.buffers = {binTexture};
    programCfg.uniforms = {
        {"index", &indexUniform},
        {"interpolate", &interpolateUniform},
        {"zoom", &zoomFactor},
        {"offset", &offsetFactor},
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

        binTexture->fill(start * input.in.buf.size() * sizeof(float), blocks * input.in.buf.size() * sizeof(float));

        start = 0;
        blocks = inc;
    }

    // TODO: Improve this.
    binTexture->fill(start * input.in.buf.size() * sizeof(float), blocks * input.in.buf.size() * sizeof(float));
    last = inc;

    zoomFactor[0] = config.zoom;
    indexUniform[0] = inc / (float)ymax;
    interpolateUniform[0] = config.interpolate;
    offsetFactor[0] = config.offset / (float)config.size.width;

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

float Generic::zoom(const float& zoom) {
    config.zoom = zoom;
    this->offset(config.offset);
    return config.zoom;
}

int Generic::offset(const int& offset) {
    config.offset = std::clamp(offset, 0,
            (int)(config.size.width - (config.size.width / config.zoom)));
    return config.offset;
}

Render::Texture& Generic::tex() const {
    return *texture;
};

} // namespace Jetstream::Waterfall
