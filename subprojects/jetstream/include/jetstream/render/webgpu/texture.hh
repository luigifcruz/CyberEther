#ifndef JETSTREAM_RENDER_WEBGPU_TEXTURE_HH
#define JETSTREAM_RENDER_WEBGPU_TEXTURE_HH

#include "jetstream/render/base/texture.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class TextureImp<Device::WebGPU> : public Texture {
 public:
    explicit TextureImp(const Config& config);

    using Render::Texture::size;
    bool size(const Size2D<U64>& size);

    Result fill();
    Result fillRow(const U64& y, const U64& height);

    void* raw() {
        return texture;
    }

 protected:
    Result create();
    Result destroy();

    constexpr wgpu::Texture& getHandle() {
        return texture;
    }

    constexpr wgpu::Sampler& getSamplerHandle() {
        return sampler;
    }

    constexpr const wgpu::TextureFormat& getTextureFormat() const {
        return textureFormat;
    }

    static wgpu::TextureFormat ConvertPixelFormat(const PixelFormat&, 
                                                  const PixelType&);
    static U64 GetPixelByteSize(const wgpu::TextureFormat&);

 private:
    wgpu::Sampler sampler;
    wgpu::Texture texture;
    wgpu::TextureFormat textureFormat;

    friend class SurfaceImp<Device::Metal>;
    friend class ProgramImp<Device::Metal>;
};

}  // namespace Jetstream::Render

#endif
