#ifndef JETSTREAM_RENDER_METAL_TEXTURE_HH
#define JETSTREAM_RENDER_METAL_TEXTURE_HH

#include "jetstream/render/base/texture.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class TextureImp<Device::Metal> : public Texture {
 public:
    explicit TextureImp(const Config& config);

    Result create();
    Result destroy();

    using Render::Texture::size;
    bool size(const Extent2D<U64>& size);

    Result fill();
    Result fillRow(const U64& y, const U64& height);

    void* raw() {
        return texture;
    }

 protected:
    constexpr MTL::PixelFormat getPixelFormat() const {
        return pixelFormat;
    }

    constexpr MTL::Texture* getHandle() const {
        return texture;
    }

    constexpr MTL::SamplerState* getSamplerStateHandle() const {
        return samplerState;
    }

    static MTL::PixelFormat ConvertPixelFormat(const PixelFormat&, 
                                               const PixelType&);
    static U64 GetPixelByteSize(const MTL::PixelFormat&);

 private:
    MTL::Texture* texture = nullptr;
    MTL::PixelFormat pixelFormat;
    MTL::SamplerState* samplerState;

    friend class SurfaceImp<Device::Metal>;
    friend class ProgramImp<Device::Metal>;
};

}  // namespace Jetstream::Render

#endif
