#include "jetstream/modules/spectrogram.hh"
#include "jetstream/render/utils.hh"

#include "resources/shaders/spectrogram_shaders.hh"
#include "jetstream/constants.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
struct Spectrogram<D, T>::GImpl {
    struct {
        U32 width;
        U32 height;
        F32 offset;
        F32 zoom;
    } signalUniforms;

    Tensor<D, F32> frequencyBins;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> signalBuffer;
    std::shared_ptr<Render::Buffer> signalUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Program> signalProgram;

    std::shared_ptr<Render::Surface> surface;

    std::shared_ptr<Render::Vertex> vertex;

    std::shared_ptr<Render::Draw> drawVertex;

    F32 decayFactor;
    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;
    U64 totalFrequencyBins = 0;
};

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
    gimpl->numberOfElements = input.buffer.shape()[last_axis];
    gimpl->numberOfBatches = (input.buffer.rank() == 2) ? input.buffer.shape()[0] : 1;
    gimpl->totalFrequencyBins = gimpl->numberOfElements * config.height;
    gimpl->decayFactor = pow(0.999, gimpl->numberOfBatches);

    // Allocate internal buffers.

    JST_CHECK(gimpl->frequencyBins.create(D, mem2::TypeToDataType<F32>(), {gimpl->numberOfElements, config.height}));

    return Result::SUCCESS;
}

template<Device D, typename T>
void Spectrogram<D, T>::info() const {
    JST_DEBUG("  Window Size: [{}, {}]", config.viewSize.x, config.viewSize.y);
    JST_DEBUG("  Height: {}", config.height);
}

template<Device D, typename T>
Result Spectrogram<D, T>::createPresent() {
    // Signal element.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenVertices;
        cfg.elementByteSize = sizeof(float);
        cfg.size = 12;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(gimpl->fillScreenVerticesBuffer, cfg));
        JST_CHECK(window->bind(gimpl->fillScreenVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenTextureVertices;
        cfg.elementByteSize = sizeof(float);
        cfg.size = 8;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(gimpl->fillScreenTextureVerticesBuffer, cfg));
        JST_CHECK(window->bind(gimpl->fillScreenTextureVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenIndices;
        cfg.elementByteSize = sizeof(uint32_t);
        cfg.size = 6;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(gimpl->fillScreenIndicesBuffer, cfg));
        JST_CHECK(window->bind(gimpl->fillScreenIndicesBuffer));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {gimpl->fillScreenVerticesBuffer, 3},
            {gimpl->fillScreenTextureVerticesBuffer, 2},
        };
        cfg.indices = gimpl->fillScreenIndicesBuffer;
        JST_CHECK(window->build(gimpl->vertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = gimpl->vertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(gimpl->drawVertex, cfg));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, gimpl->frequencyBins);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.size = gimpl->frequencyBins.size();
        cfg.elementByteSize = sizeof(F32);
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = enableZeroCopy;
        JST_CHECK(window->build(gimpl->signalBuffer, cfg));
        JST_CHECK(window->bind(gimpl->signalBuffer));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = (uint8_t*)TurboLutBytes;
        JST_CHECK(window->build(gimpl->lutTexture, cfg));
        JST_CHECK(window->bind(gimpl->lutTexture));
    }

    {
        // TODO: This could use unified memory.
        Render::Buffer::Config cfg;
        cfg.buffer = &gimpl->signalUniforms;
        cfg.elementByteSize = sizeof(gimpl->signalUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(gimpl->signalUniformBuffer, cfg));
        JST_CHECK(window->bind(gimpl->signalUniformBuffer));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["signal"];
        cfg.draws = {
            gimpl->drawVertex,
        };
        cfg.textures = {gimpl->lutTexture};
        cfg.buffers = {
            {gimpl->signalUniformBuffer, Render::Program::Target::VERTEX |
                                         Render::Program::Target::FRAGMENT},
            {gimpl->signalBuffer, Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(gimpl->signalProgram, cfg));
    }

    // Surface.

    {
        Render::Texture::Config cfg;
        cfg.size = config.viewSize;
        JST_CHECK(window->build(gimpl->framebufferTexture, cfg));
    }

    {
        Render::Surface::Config cfg;
        cfg.framebuffer = gimpl->framebufferTexture;
        cfg.programs = {gimpl->signalProgram};
        cfg.clearColor = {0.1f, 0.1f, 0.1f, 1.0f};
        JST_CHECK(window->build(gimpl->surface, cfg));
        JST_CHECK(window->bind(gimpl->surface));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(gimpl->surface));
    JST_CHECK(window->unbind(gimpl->lutTexture));
    JST_CHECK(window->unbind(gimpl->fillScreenVerticesBuffer));
    JST_CHECK(window->unbind(gimpl->fillScreenTextureVerticesBuffer));
    JST_CHECK(window->unbind(gimpl->fillScreenIndicesBuffer));
    JST_CHECK(window->unbind(gimpl->signalBuffer));
    JST_CHECK(window->unbind(gimpl->signalUniformBuffer));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::present() {
    gimpl->signalBuffer->update();

    gimpl->signalUniforms.width = gimpl->numberOfElements;
    gimpl->signalUniforms.height = config.height;
    gimpl->signalUniforms.zoom = 1.0;
    gimpl->signalUniforms.offset = 0.0;

    gimpl->signalUniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
const Extent2D<U64>& Spectrogram<D, T>::viewSize(const Extent2D<U64>& viewSize) {
    if (gimpl->surface->size(viewSize) != this->viewSize()) {
        JST_TRACE("Spectrogram size changed from [{}, {}] to [{}, {}].",
                config.viewSize.x,
                config.viewSize.y,
                viewSize.x,
                viewSize.y);

        config.viewSize = gimpl->surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Spectrogram<D, T>::getTexture() {
    return *gimpl->framebufferTexture;
};

}  // namespace Jetstream
