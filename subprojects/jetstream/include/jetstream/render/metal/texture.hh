#ifndef JETSTREAM_RENDER_METAL_TEXTURE_HH
#define JETSTREAM_RENDER_METAL_TEXTURE_HH

#include "jetstream/render/base/texture.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class TextureImp<Device::Metal> : public Texture {
 public:
    explicit TextureImp(const Config& config);

    using Render::Texture::size;
    const bool size(const Size2D<U64>& size);

    const Result fill();
    const Result fillRow(const U64& y, const U64& height);

    void* raw() {
        return texture;
    }

 protected:
    const Result create();
    const Result destroy();

    constexpr const MTL::PixelFormat getPixelFormat() const {
        return pixelFormat;
    }

    constexpr const MTL::Texture* getHandle() const {
        return texture;
    }

    static const MTL::PixelFormat ConvertPixelFormat(const PixelFormat&, 
                                                     const PixelType&);
    static const U64 GetPixelByteSize(const MTL::PixelFormat&);

 private:
    MTL::Texture* texture = nullptr;
    MTL::PixelFormat pixelFormat;

    friend class SurfaceImp<Device::Metal>;
    friend class ProgramImp<Device::Metal>;
};

}  // namespace Jetstream::Render

#endif
