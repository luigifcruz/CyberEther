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
    bufferCfg.buffer = ptr;
    bufferCfg.size = input.in.buf.size() * ymax;
    bufferCfg.elementByteSize = sizeof(float);
    binTexture = Render::Create(bufferCfg);

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)turbo_srgb_bytes;
    lutTextureCfg.key = "LutTexture";
    lutTexture = Render::Create(lutTextureCfg);

    uniformsBuffer = nonstd::span<uint8_t>((uint8_t*)&shaderUniforms, sizeof(shaderUniforms));

    Render::Program::Config programCfg;
    programCfg.shaders = {
        {Render::Backend::Metal, {MetalShader}},
        {Render::Backend::GLES, {GlesVertexShader, GlesFragmentShader}},
    };
    programCfg.draws = {drawVertex};
    programCfg.textures = {lutTexture};
    programCfg.buffers = {binTexture};
    programCfg.uniforms = {
        {"uniforms", &uniformsBuffer},
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

        binTexture->fill(start * input.in.buf.size(), blocks * input.in.buf.size());

        start = 0;
        blocks = inc;
    }

    binTexture->fill(start * input.in.buf.size(), blocks * input.in.buf.size());
    last = inc;

    shaderUniforms.zoom = config.zoom;
    shaderUniforms.width = input.in.buf.size();
    shaderUniforms.height = ymax;
    shaderUniforms.interpolate = config.interpolate;
    shaderUniforms.index = inc / (float)shaderUniforms.height;
    shaderUniforms.offset = config.offset / (float)config.size.width;
    shaderUniforms.maxSize = shaderUniforms.width * shaderUniforms.height;

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
