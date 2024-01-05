#include "jetstream/modules/waterfall.hh"
#include "shaders/waterfall_shaders.hh"

namespace Jetstream {

template<Device D, typename T>
Result Waterfall<D, T>::create() {
    JST_DEBUG("Initializing Waterfall module.");
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

    // Allocate internal buffers.

    frequencyBins = Tensor<D, F32>({numberOfElements,  config.height});

    return Result::SUCCESS;
}

template<Device D, typename T>
void Waterfall<D, T>::info() const {
    JST_INFO("  Offset:       {}", config.offset);
    JST_INFO("  Zoom:         {}", config.zoom);
    JST_INFO("  Interpolate:  {}", config.interpolate ? "YES" : "NO");
    JST_INFO("  Height:       {}", config.height);
    JST_INFO("  Window Size:  [{}, {}]", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
Result Waterfall<D, T>::createPresent() {
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

    Render::Buffer::Config bufferCfg;
    bufferCfg.buffer = frequencyBins.cpu().data();
    bufferCfg.size = frequencyBins.size();
    bufferCfg.elementByteSize = sizeof(F32);
    bufferCfg.target = Render::Buffer::Target::STORAGE;
    bufferCfg.enableZeroCopy = true;
    JST_CHECK(window->build(binTexture, bufferCfg));

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
    JST_CHECK(window->build(lutTexture, lutTextureCfg));

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
Result Waterfall<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::present() {
    int start = last;
    int blocks = (inc - last);

    // TODO: Fix this horrible thing.
    if (blocks < 0) {
        blocks = config.height - last;

        binTexture->update(start * numberOfElements, blocks * numberOfElements);

        start = 0;
        blocks = inc;
    }

    binTexture->update(start * numberOfElements, blocks * numberOfElements);
    last = inc;

    shaderUniforms.zoom = config.zoom;
    shaderUniforms.width = numberOfElements;
    shaderUniforms.height = config.height;
    shaderUniforms.interpolate = config.interpolate;
    shaderUniforms.index = inc / (float)shaderUniforms.height;
    shaderUniforms.offset = config.offset / (float)config.viewSize.width;
    shaderUniforms.maxSize = shaderUniforms.width * shaderUniforms.height;

    uniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::createCompute(const RuntimeMetadata&) {
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::compute(const RuntimeMetadata& meta) {
    auto res = underlyingCompute(meta);
    inc = (inc + numberOfBatches) % config.height;
    return res;
}

template<Device D, typename T>
const bool& Waterfall<D, T>::interpolate(const bool& val) {
    config.interpolate = val;
    return interpolate();
}

template<Device D, typename T>
const F32& Waterfall<D, T>::zoom(const F32& zoom) {
    config.zoom = zoom;
    offset(config.offset);
    return config.zoom;
}

template<Device D, typename T>
const I32& Waterfall<D, T>::offset(const I32& offset) {
    config.offset = std::clamp(offset, 0,
            (I32)(config.viewSize.width - (config.viewSize.width / config.zoom)));
    return config.offset;
}

template<Device D, typename T>
const Size2D<U64>& Waterfall<D, T>::viewSize(const Size2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        JST_DEBUG("Waterfall size changed from [{}, {}] to [{}, {}].",
                config.viewSize.width,
                config.viewSize.height,
                viewSize.width,
                viewSize.height);

        config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Waterfall<D, T>::getTexture() {
    return *texture;
};

}  // namespace Jetstream
