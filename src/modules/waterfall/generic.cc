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
    numberOfElements = input.buffer.shape()[last_axis];
    numberOfBatches = (input.buffer.rank() == 2) ? input.buffer.shape()[0] : 1;

    // Allocate internal buffers.

    frequencyBins = Tensor<D, F32>({numberOfElements,  config.height});

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
        JST_CHECK(window->build(fillScreenVerticesBuffer, cfg));
        JST_CHECK(window->bind(fillScreenVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenTextureVertices;
        cfg.elementByteSize = sizeof(float);
        cfg.size = 8;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenTextureVerticesBuffer, cfg));
        JST_CHECK(window->bind(fillScreenTextureVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenIndices;
        cfg.elementByteSize = sizeof(uint32_t);
        cfg.size = 6;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(fillScreenIndicesBuffer, cfg));
        JST_CHECK(window->bind(fillScreenIndicesBuffer));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {fillScreenVerticesBuffer, 3},
            {fillScreenTextureVerticesBuffer, 2},
        };
        cfg.indices = fillScreenIndicesBuffer;
        JST_CHECK(window->build(vertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = vertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(drawVertex, cfg));
    }

    {
        auto [buffer, enableZeroCopy] = ConvertToOptimalStorage(window, frequencyBins);

        Render::Buffer::Config cfg;
        cfg.buffer = buffer;
        cfg.size = frequencyBins.size();
        cfg.elementByteSize = sizeof(F32);
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = enableZeroCopy;
        JST_CHECK(window->build(signalBuffer, cfg));
        JST_CHECK(window->bind(signalBuffer));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = (uint8_t*)TurboLutBytes;
        JST_CHECK(window->build(lutTexture, cfg));
        JST_CHECK(window->bind(lutTexture));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &gimpl->signalUniforms;
        cfg.elementByteSize = sizeof(gimpl->signalUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(signalUniformBuffer, cfg));
        JST_CHECK(window->bind(signalUniformBuffer));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["signal"];
        cfg.draws = {
            drawVertex,
        };
        cfg.textures = {lutTexture};
        cfg.buffers = {
            {signalUniformBuffer, Render::Program::Target::VERTEX |
                            Render::Program::Target::FRAGMENT},
            {signalBuffer, Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(signalProgram, cfg));
    }

    // Surface.

    {
        Render::Texture::Config cfg;
        cfg.size = config.viewSize;
        JST_CHECK(window->build(framebufferTexture, cfg));
    }

    {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.programs = {signalProgram};
        cfg.multisampled = false;
        JST_CHECK(window->build(surface, cfg));
        JST_CHECK(window->bind(surface));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(surface));
    JST_CHECK(window->unbind(lutTexture));
    JST_CHECK(window->unbind(fillScreenVerticesBuffer));
    JST_CHECK(window->unbind(fillScreenTextureVerticesBuffer));
    JST_CHECK(window->unbind(fillScreenIndicesBuffer));
    JST_CHECK(window->unbind(signalBuffer));
    JST_CHECK(window->unbind(signalUniformBuffer));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::present() {
    int start = last;
    int blocks = (inc - last);

    // TODO: Fix this horrible thing.
    if (blocks < 0) {
        blocks = config.height - last;

        signalBuffer->update(start * numberOfElements, blocks * numberOfElements);

        start = 0;
        blocks = inc;
    }

    signalBuffer->update(start * numberOfElements, blocks * numberOfElements);
    last = inc;

    gimpl->signalUniforms.zoom = config.zoom;
    gimpl->signalUniforms.width = numberOfElements;
    gimpl->signalUniforms.height = config.height;
    gimpl->signalUniforms.interpolate = config.interpolate;
    gimpl->signalUniforms.index = inc / (float)gimpl->signalUniforms.height;
    gimpl->signalUniforms.offset = config.offset / (float)config.viewSize.x;
    gimpl->signalUniforms.maxSize = gimpl->signalUniforms.width * gimpl->signalUniforms.height;

    signalUniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::createCompute(const Context&) {
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Waterfall<D, T>::compute(const Context& ctx) {
    auto res = underlyingCompute(ctx);
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
            (I32)(config.viewSize.x - (config.viewSize.x / config.zoom)));
    return config.offset;
}

template<Device D, typename T>
const Extent2D<U64>& Waterfall<D, T>::viewSize(const Extent2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        JST_DEBUG("Waterfall size changed from [{}, {}] to [{}, {}].",
                config.viewSize.x,
                config.viewSize.y,
                viewSize.x,
                viewSize.y);

        config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Waterfall<D, T>::getTexture() {
    return *framebufferTexture;
};

}  // namespace Jetstream
