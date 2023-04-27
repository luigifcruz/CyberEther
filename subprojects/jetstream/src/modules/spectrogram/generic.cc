#include "jetstream/modules/spectrogram.hh"

namespace Jetstream {

template<Device D, typename T>
Spectrogram<D, T>::Spectrogram(const Config& config,
                               const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Spectrogram module.");
    JST_CHECK_THROW(initInput(input.buffer));
}

template<Device D, typename T>
void Spectrogram<D, T>::summary() const {
    JST_INFO("     Window Size: [{}, {}]", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
Result Spectrogram<D, T>::createPresent(Render::Window& window) {
    Render::Buffer::Config fillScreenVerticesConf;
    fillScreenVerticesConf.buffer = &Render::Extras::FillScreenVertices;
    fillScreenVerticesConf.elementByteSize = sizeof(float);
    fillScreenVerticesConf.size = 12;
    fillScreenVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(window.build(fillScreenVerticesBuffer, fillScreenVerticesConf));

    Render::Buffer::Config fillScreenTextureVerticesConf;
    fillScreenTextureVerticesConf.buffer = &Render::Extras::FillScreenTextureVertices;
    fillScreenTextureVerticesConf.elementByteSize = sizeof(float);
    fillScreenTextureVerticesConf.size = 8;
    fillScreenTextureVerticesConf.target = Render::Buffer::Target::VERTEX;
    JST_CHECK(window.build(fillScreenTextureVerticesBuffer, fillScreenTextureVerticesConf));

    Render::Buffer::Config fillScreenIndicesConf;
    fillScreenIndicesConf.buffer = &Render::Extras::FillScreenIndices;
    fillScreenIndicesConf.elementByteSize = sizeof(uint32_t);
    fillScreenIndicesConf.size = 6;
    fillScreenIndicesConf.target = Render::Buffer::Target::VERTEX_INDICES;
    JST_CHECK(window.build(fillScreenIndicesBuffer, fillScreenIndicesConf));

    Render::Vertex::Config vertexCfg;
    vertexCfg.buffers = {
        {fillScreenVerticesBuffer, 3},
        {fillScreenTextureVerticesBuffer, 2},
    };
    vertexCfg.indices = fillScreenIndicesBuffer;
    JST_CHECK(window.build(vertex, vertexCfg));

    Render::Draw::Config drawVertexCfg;
    drawVertexCfg.buffer = vertex;
    drawVertexCfg.mode = Render::Draw::Mode::TRIANGLES;
    JST_CHECK(window.build(drawVertex, drawVertexCfg));

    Render::Buffer::Config bufferCfg;
    bufferCfg.buffer = frequencyBins.data();
    bufferCfg.size = frequencyBins.size();
    bufferCfg.elementByteSize = sizeof(frequencyBins[0]);
    bufferCfg.target = Render::Buffer::Target::STORAGE;
    bufferCfg.enableZeroCopy = true;
    JST_CHECK(window.build(binTexture, bufferCfg));

    Render::Texture::Config lutTextureCfg;
    lutTextureCfg.size = {256, 1};
    lutTextureCfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
    lutTextureCfg.key = "LutTexture";
    JST_CHECK(window.build(lutTexture, lutTextureCfg));

    // TODO: This could use unified memory.
    Render::Buffer::Config uniformCfg;
    uniformCfg.buffer = &shaderUniforms;
    uniformCfg.elementByteSize = sizeof(shaderUniforms);
    uniformCfg.size = 1;
    uniformCfg.target = Render::Buffer::Target::STORAGE;
    JST_CHECK(window.build(uniformBuffer, uniformCfg));

    Render::Program::Config programCfg;
    programCfg.shaders = {
        {Device::Metal, {MetalShader}},
    };
    programCfg.draws = {drawVertex};
    programCfg.textures = {lutTexture};
    programCfg.buffers = {uniformBuffer, binTexture};
    JST_CHECK(window.build(program, programCfg));

    Render::Texture::Config textureCfg;
    textureCfg.size = config.viewSize;
    JST_CHECK(window.build(texture, textureCfg));

    Render::Surface::Config surfaceCfg;
    surfaceCfg.framebuffer = texture;
    surfaceCfg.programs = {program};
    JST_CHECK(window.build(surface, surfaceCfg));
    JST_CHECK(window.bind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::present(Render::Window& window) {
    binTexture->update();

    shaderUniforms.width = input.buffer.shape(1);
    shaderUniforms.height = config.height;
    shaderUniforms.zoom = 1.0;
    shaderUniforms.offset = 0.0;
    uniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
const Render::Size2D<U64>& Spectrogram<D, T>::viewSize(const Render::Size2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        JST_TRACE("Spectrogram size changed from [{}, {}] to [{}, {}].", 
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
