#include "jetstream/modules/constellation.hh"
#include "shaders/constellation_shaders.hh"

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
};

template<Device D, typename T>
Result Constellation<D, T>::create() {
    JST_DEBUG("Initializing Constellation module.");
    JST_INIT_IO();

    // Allocate internal buffers.

    timeSamples = Tensor<D, F32>({config.viewSize.x, config.viewSize.y});

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
        cfg.buffer = &Render::Extras::FillScreenVertices;
        cfg.elementByteSize = sizeof(float);
        cfg.size = 12;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenVerticesBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &Render::Extras::FillScreenTextureVertices;
        cfg.elementByteSize = sizeof(float);
        cfg.size = 8;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenTextureVerticesBuffer, cfg));
    }
    
    {
        Render::Buffer::Config cfg;
        cfg.buffer = &Render::Extras::FillScreenIndices;
        cfg.elementByteSize = sizeof(uint32_t);
        cfg.size = 6;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(fillScreenIndicesBuffer, cfg));
    }
    
    {
        Render::Vertex::Config cfg;
        cfg.buffers = {
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
        Render::Texture::Config cfg;
        cfg.buffer = (U8*)(timeSamples.data());
        cfg.size = {timeSamples.shape()[0], timeSamples.shape()[1]};
        cfg.dfmt = Render::Texture::DataFormat::F32;
        cfg.pfmt = Render::Texture::PixelFormat::RED;
        cfg.ptype = Render::Texture::PixelType::F32;
        JST_CHECK(window->build(signalTexture, cfg));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = (uint8_t*)Render::Extras::TurboLutBytes;
        JST_CHECK(window->build(lutTexture, cfg));
    }
    
    {
        // TODO: This could use unified memory.
        Render::Buffer::Config cfg;
        cfg.buffer = &gimpl->signalUniforms;
        cfg.elementByteSize = sizeof(gimpl->signalUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::STORAGE;
        JST_CHECK(window->build(signalUniformBuffer, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["signal"];
        cfg.draw = drawVertex;
        cfg.textures = {signalTexture, lutTexture};
        cfg.buffers = {
            {signalUniformBuffer, Render::Program::Target::VERTEX |
                                  Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(program, cfg));
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
        cfg.programs = {program};
        cfg.buffers = {
            fillScreenVerticesBuffer,
            fillScreenTextureVerticesBuffer,
            fillScreenIndicesBuffer,
            signalUniformBuffer,
        };
        JST_CHECK(window->build(surface, cfg));
        JST_CHECK(window->bind(surface));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::present() {
    signalTexture->fill();

    gimpl->signalUniforms.width = timeSamples.shape()[0];
    gimpl->signalUniforms.height = timeSamples.shape()[1];
    gimpl->signalUniforms.zoom = 1.0;
    gimpl->signalUniforms.offset = 0.0;

    signalUniformBuffer->update();

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(surface));

    return Result::SUCCESS;
}

template<Device D, typename T>
const Extent2D<U64>& Constellation<D, T>::viewSize(const Extent2D<U64>& viewSize) {
    if (surface->size(viewSize) != this->viewSize()) {
        JST_TRACE("Constellation size changed from [{}, {}] to [{}, {}].",
                config.viewSize.x,
                config.viewSize.y,
                viewSize.x,
                viewSize.y);

        config.viewSize = surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Constellation<D, T>::getTexture() {
    return *framebufferTexture;
};

}  // namespace Jetstream
