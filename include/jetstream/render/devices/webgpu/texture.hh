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
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(textureView));
    }

 protected:
    constexpr WGPUTexture getHandle() const {
        return texture;
    }

    constexpr WGPUSampler getSamplerHandle() const {
        return sampler;
    }

    constexpr WGPUTextureView getViewHandle() const {
        return textureView;
    }

    constexpr const WGPUTextureFormat& getTextureFormat() const {
        return textureFormat;
    }

    constexpr const WGPUTextureBindingLayout& getTextureBindingLayout() const {
        return textureBindingLayout;
    }

    constexpr const WGPUSamplerBindingLayout& getSamplerBindingLayout() const {
        return samplerBindingLayout;
    }

    static WGPUTextureFormat ConvertPixelFormat(const PixelFormat&,
                                                const PixelType&);
    static U64 GetPixelByteSize(const WGPUTextureFormat&);

 private:
    WGPUTexture texture;
    WGPUTextureView textureView;
    WGPUSampler sampler;
    WGPUTextureFormat textureFormat;
    WGPUSamplerBindingLayout samplerBindingLayout;
    WGPUTextureBindingLayout textureBindingLayout;

    friend class SurfaceImp<Device::WebGPU>;
    friend class ProgramImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
