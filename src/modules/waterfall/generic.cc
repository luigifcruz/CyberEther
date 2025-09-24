#include "jetstream/modules/waterfall.hh"
#include "jetstream/render/utils.hh"

#include "resources/shaders/waterfall_shaders.hh"
#include "jetstream/constants.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
struct Waterfall<D, T>::GImpl {
    struct {
        int width;
        int height;
        int maxSize;
        float index;
        float offset;
        float zoom;
        bool interpolate;
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

    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;
    int inc = 0, last = 0, ymax = 0;

    Result underlyingCompute(Waterfall<D, T>& m, const Context& ctx);
};

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
    gimpl->numberOfElements = input.buffer.shape()[last_axis];
    gimpl->numberOfBatches = (input.buffer.rank() == 2) ? input.buffer.shape()[0] : 1;

    // Allocate internal buffers.

    JST_CHECK(gimpl->frequencyBins.create(D, mem2::TypeToDataType<F32>(), {gimpl->numberOfElements, config.height}));

    return Result::SUCCESS;
}

template<Device D, typename T>
void Waterfall<D, T>::info() const {
    JST_DEBUG("  Offset:       {}", config.offset);
    JST_DEBUG("  Zoom:         {}", config.zoom);
    JST_DEBUG("  Interpolate:  {}", config.interpolate ? "YES" : "NO");
    JST_DEBUG("  Height:       {}", config.height);
    JST_DEBUG("  Window Size:  [{}, {}]", config.viewSize.x, config.viewSize.y);
}

template<Device D, typename T>
Result Waterfall<D, T>::createPresent() {
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
        cfg.multisampled = false;
        JST_CHECK(window->build(gimpl->surface, cfg));
        JST_CHECK(window->bind(gimpl->surface));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::destroyPresent() {
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
Result Waterfall<D, T>::present() {
    int start = gimpl->last;
    int blocks = (gimpl->inc - gimpl->last);

    // TODO: Fix this horrible thing.
    if (blocks < 0) {
        blocks = config.height - gimpl->last;

        gimpl->signalBuffer->update(start * gimpl->numberOfElements, blocks * gimpl->numberOfElements);

        start = 0;
        blocks = gimpl->inc;
    }

    gimpl->signalBuffer->update(start * gimpl->numberOfElements, blocks * gimpl->numberOfElements);
    gimpl->last = gimpl->inc;

    gimpl->signalUniforms.zoom = config.zoom;
    gimpl->signalUniforms.width = gimpl->numberOfElements;
    gimpl->signalUniforms.height = config.height;
    gimpl->signalUniforms.interpolate = config.interpolate;
    gimpl->signalUniforms.index = gimpl->inc / (float)gimpl->signalUniforms.height;
    gimpl->signalUniforms.offset = config.offset / (float)config.viewSize.x;
    gimpl->signalUniforms.maxSize = gimpl->signalUniforms.width * gimpl->signalUniforms.height;

    gimpl->signalUniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::createCompute(const Context&) {
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::compute(const Context& ctx) {
    auto res = gimpl->underlyingCompute(*this, ctx);
    gimpl->inc = (gimpl->inc + gimpl->numberOfBatches) % config.height;
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
            (I32)(config.viewSize.x - (config.viewSize.x / config.zoom)));
    return config.offset;
}

template<Device D, typename T>
const Extent2D<U64>& Waterfall<D, T>::viewSize(const Extent2D<U64>& viewSize) {
    if (gimpl->surface->size(viewSize) != this->viewSize()) {
        JST_DEBUG("Waterfall size changed from [{}, {}] to [{}, {}].",
                config.viewSize.x,
                config.viewSize.y,
                viewSize.x,
                viewSize.y);

        config.viewSize = gimpl->surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Waterfall<D, T>::getTexture() {
    return *gimpl->framebufferTexture;
};

}  // namespace Jetstream
