#include "jetstream/modules/constellation.hh"

#include "resources/shaders/constellation_shaders.hh"
#include "jetstream/constants.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
struct Constellation<D, T>::GImpl {
    struct {
        U32 width;
        U32 height;
        F32 offset;
        F32 zoom;
    } signalUniforms;

    Tensor<D, F32> timeSamples;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> signalUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Texture> signalTexture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Program> program;

    std::shared_ptr<Render::Surface> surface;

    std::shared_ptr<Render::Vertex> vertex;

    std::shared_ptr<Render::Draw> drawVertex;

    F32 decayFactor;
};

template<Device D, typename T>
Result Constellation<D, T>::create() {
    JST_DEBUG("Initializing Constellation module.");
    JST_INIT_IO();

    // Check input parameters.

    if (input.buffer.rank() == 0) {
        JST_ERROR("Input buffer rank is 0.");
        return Result::ERROR;
    }

    if (input.buffer.size() == 0) {
        JST_ERROR("Input is empty during initialization.");
        return Result::ERROR;
    }

    // Allocate internal buffers.

    gimpl->timeSamples = Tensor<D, F32>({config.viewSize.x, config.viewSize.y});

    return Result::SUCCESS;
}

template<Device D, typename T>
void Constellation<D, T>::info() const {
    JST_DEBUG("  Window Size: [{}, {}]", config.viewSize.x, config.viewSize.y);
}

template<Device D, typename T>
Result Constellation<D, T>::createPresent() {
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
        Render::Texture::Config cfg;
        cfg.buffer = (U8*)(gimpl->timeSamples.data());
        cfg.size = {gimpl->timeSamples.shape()[0], gimpl->timeSamples.shape()[1]};
        cfg.dfmt = Render::Texture::DataFormat::F32;
        cfg.pfmt = Render::Texture::PixelFormat::RED;
        cfg.ptype = Render::Texture::PixelType::F32;
        JST_CHECK(window->build(gimpl->signalTexture, cfg));
        JST_CHECK(window->bind(gimpl->signalTexture));
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
        cfg.target = Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(gimpl->signalUniformBuffer, cfg));
        JST_CHECK(window->bind(gimpl->signalUniformBuffer));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["signal"];
        cfg.draws = {
            gimpl->drawVertex,
        };
        cfg.textures = {gimpl->signalTexture, gimpl->lutTexture};
        cfg.buffers = {
            {gimpl->signalUniformBuffer, Render::Program::Target::VERTEX |
                                         Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(gimpl->program, cfg));
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
        cfg.programs = {gimpl->program};
        cfg.clearColor = {0.1f, 0.1f, 0.1f, 1.0f};
        JST_CHECK(window->build(gimpl->surface, cfg));
        JST_CHECK(window->bind(gimpl->surface));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::present() {
    gimpl->signalTexture->fill();

    gimpl->signalUniforms.width = gimpl->timeSamples.shape()[0];
    gimpl->signalUniforms.height = gimpl->timeSamples.shape()[1];
    gimpl->signalUniforms.zoom = 1.0;
    gimpl->signalUniforms.offset = 0.0;

    gimpl->signalUniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(gimpl->surface));
    JST_CHECK(window->unbind(gimpl->signalTexture));
    JST_CHECK(window->unbind(gimpl->lutTexture));
    JST_CHECK(window->unbind(gimpl->fillScreenVerticesBuffer));
    JST_CHECK(window->unbind(gimpl->fillScreenTextureVerticesBuffer));
    JST_CHECK(window->unbind(gimpl->fillScreenIndicesBuffer));
    JST_CHECK(window->unbind(gimpl->signalUniformBuffer));

    return Result::SUCCESS;
}

template<Device D, typename T>
const Extent2D<U64>& Constellation<D, T>::viewSize(const Extent2D<U64>& viewSize) {
    if (gimpl->surface->size(viewSize) != this->viewSize()) {
        JST_TRACE("Constellation size changed from [{}, {}] to [{}, {}].",
                config.viewSize.x,
                config.viewSize.y,
                viewSize.x,
                viewSize.y);

        config.viewSize = gimpl->surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Constellation<D, T>::getTexture() {
    return *gimpl->framebufferTexture;
};

}  // namespace Jetstream
