#include "jetstream/modules/waterfall.hh"
#include "shaders/waterfall_shaders.hh"

namespace Jetstream {

template<Device D, typename T>
Waterfall<D, T>::Waterfall(const Config& config, 
                           const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Waterfall module.");

    JST_CHECK_THROW(Module::initInput(input.buffer));
}

template<Device D, typename T>
void Waterfall<D, T>::summary() const {
    JST_INFO("    Offset:       {}", config.offset);
    JST_INFO("    Zoom:         {}", config.zoom);
    JST_INFO("    Interpolate:  {}", config.interpolate ? "YES" : "NO");
    JST_INFO("    Height:       {}", config.height);
    JST_INFO("    Window Size:  [{}, {}]", config.viewSize.width, config.viewSize.height);
}

template<Device D, typename T>
Result Waterfall<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Waterfall compute core.");

    frequencyBins = Vector<D, F32, 2>({input.buffer.shape()[1],  config.height});

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::createPresent(Render::Window& window) {
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
    JST_CHECK(window.build(lutTexture, lutTextureCfg));

    Render::Buffer::Config uniformCfg;
    uniformCfg.buffer = &shaderUniforms;
    uniformCfg.elementByteSize = sizeof(shaderUniforms);
    uniformCfg.size = 1;
    uniformCfg.target = Render::Buffer::Target::UNIFORM;
    JST_CHECK(window.build(uniformBuffer, uniformCfg));

    Render::Program::Config programCfg;
    programCfg.shaders = {
        {Device::Metal,  {signal_msl_vert_shader, signal_msl_frag_shader}},
        {Device::Vulkan, {signal_spv_vert_shader, signal_spv_frag_shader}},
    };
    programCfg.draw = drawVertex;
    programCfg.textures = {lutTexture};
    programCfg.buffers = {
        {uniformBuffer, Render::Program::Target::VERTEX |
                        Render::Program::Target::FRAGMENT},
        {binTexture, Render::Program::Target::FRAGMENT},
    };
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
Result Waterfall<D, T>::compute(const RuntimeMetadata& meta) {
    auto res = this->underlyingCompute(meta);
    inc = (inc + input.buffer.shape()[0]) % config.height;
    return res;
}

template<Device D, typename T>
Result Waterfall<D, T>::present(Render::Window&) {
    int start = last;
    int blocks = (inc - last);

    // TODO: Fix this horrible thing.
    if (blocks < 0) {
        blocks = config.height - last;

        binTexture->update(start * input.buffer.shape()[1], blocks * input.buffer.shape()[1]);

        start = 0;
        blocks = inc;
    }

    binTexture->update(start * input.buffer.shape()[1], blocks * input.buffer.shape()[1]);
    last = inc;

    shaderUniforms.zoom = config.zoom;
    shaderUniforms.width = input.buffer.shape()[1];
    shaderUniforms.height = config.height;
    shaderUniforms.interpolate = config.interpolate;
    shaderUniforms.index = inc / (float)shaderUniforms.height;
    shaderUniforms.offset = config.offset / (float)config.viewSize.width;
    shaderUniforms.maxSize = shaderUniforms.width * shaderUniforms.height;

    uniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
const bool& Waterfall<D, T>::interpolate(const bool& val) {
    config.interpolate = val;
    return this->interpolate();
}

template<Device D, typename T>
const F32& Waterfall<D, T>::zoom(const F32& zoom) {
    config.zoom = zoom;
    this->offset(config.offset);
    return config.zoom;
}

template<Device D, typename T>
const I32& Waterfall<D, T>::offset(const I32& offset) {
    config.offset = std::clamp(offset, 0,
            (I32)(config.viewSize.width - (config.viewSize.width / config.zoom)));
    return config.offset;
}

template<Device D, typename T>
const Render::Size2D<U64>& Waterfall<D, T>::viewSize(const Render::Size2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        JST_DEBUG("Waterfall size changed from [{}, {}] to [{}, {}].", 
                config.viewSize.width, 
                config.viewSize.height, 
                viewSize.width, 
                viewSize.height);

        this->config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Waterfall<D, T>::getTexture() {
    return *texture;
};

template<Device D, typename T>
Result Waterfall<D, T>::Factory(std::unordered_map<std::string, std::any>& configMap,
                                std::unordered_map<std::string, std::any>& inputMap,
                                std::unordered_map<std::string, std::any>&,
                                std::shared_ptr<Waterfall<D, T>>& module) {
    using Module = Waterfall<D, T>;

    Module::Config config{};

    JST_CHECK(Module::BindVariable(configMap, "zoom", config.zoom));
    JST_CHECK(Module::BindVariable(configMap, "offset", config.offset));
    JST_CHECK(Module::BindVariable(configMap, "height", config.height));
    JST_CHECK(Module::BindVariable(configMap, "interpolate", config.interpolate));
    JST_CHECK(Module::BindVariable(configMap, "viewSize", config.viewSize));

    Module::Input input{};

    JST_CHECK(Module::BindVariable(inputMap, "buffer", input.buffer));

    module = std::make_shared<Module>(config, input);

    return Result::SUCCESS;
}

}  // namespace Jetstream
