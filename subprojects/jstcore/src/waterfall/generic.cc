#include "jstcore/waterfall/shaders.hpp"
#include "jstcore/waterfall/generic.hpp"

namespace Jetstream::Waterfall {

Generic::Generic(const Config& config, const Input& input) : config(config), input(input) {}

Result Generic::initRender(uint8_t* ptr, bool cudaInterop) {
    Render::Buffer::Config fillScreenVerticesConf;
    fillScreenVerticesConf.buffer = &Render::Extras::FillScreenVertices;
    fillScreenVerticesConf.elementByteSize = sizeof(float);
    fillScreenVerticesConf.size = 12;
    fillScreenVerticesConf.target = Render::Buffer::Target::VERTEX;
    fillScreenVerticesBuffer = Render::Create(fillScreenVerticesConf);

    Render::Buffer::Config fillScreenTextureVerticesConf;
    fillScreenTextureVerticesConf.buffer = &Render::Extras::FillScreenTextureVertices;
    fillScreenTextureVerticesConf.elementByteSize = sizeof(float);
    fillScreenTextureVerticesConf.size = 8;
    fillScreenTextureVerticesConf.target = Render::Buffer::Target::VERTEX;
    fillScreenTextureVerticesBuffer = Render::Create(fillScreenTextureVerticesConf);

    Render::Buffer::Config fillScreenIndicesConf;
    fillScreenIndicesConf.buffer = &Render::Extras::FillScreenIndices;
    fillScreenIndicesConf.elementByteSize = sizeof(uint32_t);
    fillScreenIndicesConf.size = 6;
    fillScreenIndicesConf.target = Render::Buffer::Target::VERTEX_INDICES;
    fillScreenIndicesBuffer = Render::Create(fillScreenIndicesConf);

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = {
        {fillScreenVerticesBuffer, 3},
        {fillScreenTextureVerticesBuffer, 2},
    };
    vertexCfg.indices = fillScreenIndicesBuffer;
    vertex = Render::Create(vertexCfg);

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Triangles;
    drawVertex = Render::Create(drawVertexCfg);

    Render::Buffer::Config bufferCfg;
    bufferCfg.buffer = ptr;
    bufferCfg.size = input.in.buf.size() * ymax;
    bufferCfg.elementByteSize = sizeof(float);
    bufferCfg.target = Render::Buffer::Target::STORAGE;
    binTexture = Render::Create(bufferCfg);

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)turbo_srgb_bytes;
    lutTextureCfg.key = "LutTexture";
    lutTexture = Render::Create(lutTextureCfg);

    Render::Buffer::Config uniformCfg;
    uniformCfg.buffer = &shaderUniforms;
    uniformCfg.elementByteSize = sizeof(shaderUniforms);
    uniformCfg.size = 1;
    uniformCfg.target = Render::Buffer::Target::STORAGE;
    uniformBuffer = Render::Create(uniformCfg);

    Render::Program::Config programCfg;
    programCfg.shaders = {
        {Render::Backend::Metal, {MetalShader}},
        {Render::Backend::GLES, {GlesVertexShader, GlesFragmentShader}},
    };
    programCfg.draws = {drawVertex};
    programCfg.textures = {lutTexture};
    programCfg.buffers = {uniformBuffer, binTexture};
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

const Result Generic::compute() {
    auto res = this->underlyingCompute();
    inc = (inc + 1) % ymax;
    return res;
}

const Result Generic::present() {
    int start = last;
    int blocks = (inc - last);

    // TODO: Fix this horrible thing.
    if (blocks < 0) {
        blocks = ymax - last;

        binTexture->update(start * input.in.buf.size(), blocks * input.in.buf.size());

        start = 0;
        blocks = inc;
    }

    binTexture->update(start * input.in.buf.size(), blocks * input.in.buf.size());
    last = inc;

    shaderUniforms.zoom = config.zoom;
    shaderUniforms.width = input.in.buf.size();
    shaderUniforms.height = ymax;
    shaderUniforms.interpolate = config.interpolate;
    shaderUniforms.index = inc / (float)shaderUniforms.height;
    shaderUniforms.offset = config.offset / (float)config.size.width;
    shaderUniforms.maxSize = shaderUniforms.width * shaderUniforms.height;

    uniformBuffer->update();

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
