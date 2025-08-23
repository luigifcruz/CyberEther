#ifndef JETSTREAM_RENDER_WEBGPU_TEXTURE_HH
#define JETSTREAM_RENDER_WEBGPU_TEXTURE_HH

#include "jetstream/render/base/texture.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class TextureImp<Device::WebGPU> : public Texture {
 public:
    explicit TextureImp(const Config& config);

    Result create();
    Result destroy();

    using Render::Texture::size;
    bool size(const Extent2D<U64>& size);

    Result fill();
    Result fillRow(const U64& y, const U64& height);

    uint64_t raw() {
        return textureView.Get();
    }

 protected:
    constexpr wgpu::Texture& getHandle() {
        return texture;
    }

    constexpr wgpu::Sampler& getSamplerHandle() {
        return sampler;
    }

    constexpr wgpu::TextureView& getViewHandle() {
        return textureView;
    }

    constexpr const wgpu::TextureFormat& getTextureFormat() const {
        return textureFormat;
    }

    constexpr const wgpu::TextureBindingLayout& getTextureBindingLayout() const {
        return textureBindingLayout;
    }

    constexpr const wgpu::SamplerBindingLayout& getSamplerBindingLayout() const {
        return samplerBindingLayout;
    }

    static wgpu::TextureFormat ConvertPixelFormat(const PixelFormat&,
                                                  const PixelType&);
    static U64 GetPixelByteSize(const wgpu::TextureFormat&);

 private:
    wgpu::Texture texture;
    wgpu::TextureView textureView;
    wgpu::Sampler sampler;
    wgpu::TextureFormat textureFormat;
    wgpu::SamplerBindingLayout samplerBindingLayout;
    wgpu::TextureBindingLayout textureBindingLayout;

    friend class SurfaceImp<Device::WebGPU>;
    friend class ProgramImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
