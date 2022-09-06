#include "jetstream/modules/spectrogram.hh"

namespace Jetstream {

template<Device D, typename T>
const Result Spectrogram<D, T>::initializeRender() {
    Render::Buffer::Config fillScreenVerticesConf;
    fillScreenVerticesConf.buffer = &Render::Extras::FillScreenVertices;
    fillScreenVerticesConf.elementByteSize = sizeof(float);
    fillScreenVerticesConf.size = 12;
    fillScreenVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(Render::Create(fillScreenVerticesBuffer, fillScreenVerticesConf));

    Render::Buffer::Config fillScreenTextureVerticesConf;
    fillScreenTextureVerticesConf.buffer = &Render::Extras::FillScreenTextureVertices;
    fillScreenTextureVerticesConf.elementByteSize = sizeof(float);
    fillScreenTextureVerticesConf.size = 8;
    fillScreenTextureVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(Render::Create(fillScreenTextureVerticesBuffer, fillScreenTextureVerticesConf));

    Render::Buffer::Config fillScreenIndicesConf;
    fillScreenIndicesConf.buffer = &Render::Extras::FillScreenIndices;
    fillScreenIndicesConf.elementByteSize = sizeof(uint32_t);
    fillScreenIndicesConf.size = 6;
    fillScreenIndicesConf.target = Render::Buffer::Target::VERTEX_INDICES;
    JST_CHECK(Render::Create(fillScreenIndicesBuffer, fillScreenIndicesConf));

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = {
        {fillScreenVerticesBuffer, 3},
        {fillScreenTextureVerticesBuffer, 2},
    };
    vertexCfg.indices = fillScreenIndicesBuffer;
    JST_CHECK(Render::Create(vertex, vertexCfg));

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Mode::TRIANGLES;
    JST_CHECK(Render::Create(drawVertex, drawVertexCfg));

    Render::Buffer::Config bufferCfg;
    bufferCfg.buffer = frequencyBins.data();
    bufferCfg.size = frequencyBins.size();
    bufferCfg.elementByteSize = sizeof(frequencyBins[0]);
    bufferCfg.target = Render::Buffer::Target::STORAGE;
    JST_CHECK(Render::Create(binTexture, bufferCfg));

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
    lutTextureCfg.key = "LutTexture";
    JST_CHECK(Render::Create(lutTexture, lutTextureCfg));

    Render::Buffer::Config uniformCfg;
    uniformCfg.buffer = &shaderUniforms;
    uniformCfg.elementByteSize = sizeof(shaderUniforms);
    uniformCfg.size = 1;
    uniformCfg.target = Render::Buffer::Target::STORAGE;
    JST_CHECK(Render::Create(uniformBuffer, uniformCfg));

    Render::Program::Config programCfg;
    programCfg.shaders = {
        {Device::Metal, {MetalShader}},
    };
    programCfg.draws = {drawVertex};
    programCfg.textures = {lutTexture};
    programCfg.buffers = {uniformBuffer, binTexture};
    JST_CHECK(Render::Create(program, programCfg));

    Render::Texture::Config textureCfg;
    textureCfg.size = config.viewSize;
    JST_CHECK(Render::Create(texture, textureCfg));

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    JST_CHECK(Render::Create(surface, surfaceCfg));

    return Result::SUCCESS;
}

template<Device D, typename T>
Spectrogram<D, T>::Spectrogram(const Config& config, const Input& input) 
    : config(config), input(input) {
    JST_DEBUG("Initializing Spectrogram module.");

    JST_CHECK_THROW(underlyingInitialize());
    JST_CHECK_THROW(initializeRender());

    JST_INFO("===== Spectrogram Module Configuration");
    JST_INFO("Height: {}", config.height);
    JST_INFO("Window Size: {}x{}", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
const Result Spectrogram<D, T>::compute(const RuntimeMetadata& meta) {
    auto res = this->underlyingCompute();
    inc = (inc + 1) % config.height;
    return res;
}

template<Device D, typename T>
const Result Spectrogram<D, T>::present(const RuntimeMetadata& meta) {
    binTexture->update(0, frequencyBins.size());

    shaderUniforms.zoom = 1.0;
    shaderUniforms.width = input.buffer.size();
    shaderUniforms.height = config.height;
    shaderUniforms.interpolate = true;
    shaderUniforms.index = 0.0 / (float)shaderUniforms.height;
    shaderUniforms.offset = 0.0 / (float)config.viewSize.width;
    shaderUniforms.maxSize = shaderUniforms.width * shaderUniforms.height;

    uniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
const Render::Size2D<U64>& Spectrogram<D, T>::viewSize(const Render::Size2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        JST_DEBUG("Spectrogram size changed from {}x{} to {}x{}.", 
                config.viewSize.width, 
                config.viewSize.height, 
                viewSize.width, 
                viewSize.height);

        this->config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Spectrogram<D, T>::getTexture() {
    return *texture;
};

}  // namespace Jetstream
