#include "jetstream/modules/spectrogram.hh"
#include "shaders/spectrogram_shaders.hh"
#include "jetstream/render/utils.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result Spectrogram<D, T>::create() {
    JST_DEBUG("Initializing Spectrogram module.");
    JST_INIT_IO();

    // Check parameters.

    if (input.buffer.rank() > 2) {
        JST_ERROR("Invalid input rank ({}). It should be `1` or `2`.", input.buffer.rank());
        return Result::ERROR;
    }

    // Calculate parameters.

    const U64 last_axis = input.buffer.rank() - 1;
    numberOfElements = input.buffer.shape()[last_axis];
    numberOfBatches = (input.buffer.rank() == 2) ? input.buffer.shape()[0] : 1;
    totalFrequencyBins = numberOfElements * config.height;
    decayFactor = pow(0.999, numberOfBatches);

    // Allocate internal buffers.

    frequencyBins = Tensor<D, F32>({numberOfElements, config.height});

    return Result::SUCCESS;
}

template<Device D, typename T>
void Spectrogram<D, T>::info() const {
    JST_DEBUG("  Window Size: [{}, {}]", config.viewSize.width, config.viewSize.height);
    JST_DEBUG("  Height: {}", config.height);
}

template<Device D, typename T>
Result Spectrogram<D, T>::createPresent() {
    Render::Buffer::Config fillScreenVerticesConf;
    fillScreenVerticesConf.buffer = &Render::Extras::FillScreenVertices;
    fillScreenVerticesConf.elementByteSize = sizeof(float);
    fillScreenVerticesConf.size = 12;
    fillScreenVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(window->build(fillScreenVerticesBuffer, fillScreenVerticesConf));

    Render::Buffer::Config fillScreenTextureVerticesConf;
    fillScreenTextureVerticesConf.buffer = &Render::Extras::FillScreenTextureVertices;
    fillScreenTextureVerticesConf.elementByteSize = sizeof(float);
    fillScreenTextureVerticesConf.size = 8;
    fillScreenTextureVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(window->build(fillScreenTextureVerticesBuffer, fillScreenTextureVerticesConf));

    Render::Buffer::Config fillScreenIndicesConf;
    fillScreenIndicesConf.buffer = &Render::Extras::FillScreenIndices;
    fillScreenIndicesConf.elementByteSize = sizeof(uint32_t);
    fillScreenIndicesConf.size = 6;
    fillScreenIndicesConf.target = Render::Buffer::Target::VERTEX_INDICES;
    JST_CHECK(window->build(fillScreenIndicesBuffer, fillScreenIndicesConf));

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = {
        {fillScreenVerticesBuffer, 3},
        {fillScreenTextureVerticesBuffer, 2},
    };
    vertexCfg.indices = fillScreenIndicesBuffer;
    JST_CHECK(window->build(vertex, vertexCfg));

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Mode::TRIANGLES;
    JST_CHECK(window->build(drawVertex, drawVertexCfg));

    auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, frequencyBins);

    Render::Buffer::Config bufferCfg;
    bufferCfg.buffer = buffer;
    bufferCfg.size = frequencyBins.size();
    bufferCfg.elementByteSize = sizeof(F32);
    bufferCfg.target = Render::Buffer::Target::STORAGE;
    bufferCfg.enableZeroCopy = enableZeroCopy;
    JST_CHECK(window->build(binTexture, bufferCfg));

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
    JST_CHECK(window->build(lutTexture, lutTextureCfg));

    // TODO: This could use unified memory.
    Render::Buffer::Config uniformCfg;
    uniformCfg.buffer = &shaderUniforms;
    uniformCfg.elementByteSize = sizeof(shaderUniforms);
    uniformCfg.size = 1;
    uniformCfg.target = Render::Buffer::Target::UNIFORM;
    JST_CHECK(window->build(uniformBuffer, uniformCfg));

    Render::Program::Config programCfg;
    programCfg.shaders = ShadersPackage["signal"];
    programCfg.draw = drawVertex;
    programCfg.textures = {lutTexture};
    programCfg.buffers = {
        {uniformBuffer, Render::Program::Target::VERTEX |
                        Render::Program::Target::FRAGMENT},
        {binTexture, Render::Program::Target::FRAGMENT},
    };
    JST_CHECK(window->build(program, programCfg));

    Render::Texture::Config textureCfg;
    textureCfg.size = config.viewSize;
    JST_CHECK(window->build(texture, textureCfg));

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    JST_CHECK(window->build(surface, surfaceCfg));
    JST_CHECK(window->bind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::present() {
    binTexture->update();

    shaderUniforms.width = numberOfElements;
    shaderUniforms.height = config.height;
    shaderUniforms.zoom = 1.0;
    shaderUniforms.offset = 0.0;
    uniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
const Size2D<U64>& Spectrogram<D, T>::viewSize(const Size2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        JST_TRACE("Spectrogram size changed from [{}, {}] to [{}, {}].",
                config.viewSize.width,
                config.viewSize.height,
                viewSize.width,
                viewSize.height);

        config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Spectrogram<D, T>::getTexture() {
    return *texture;
};

}  // namespace Jetstream
