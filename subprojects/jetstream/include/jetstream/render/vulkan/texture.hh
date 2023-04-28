#ifndef JETSTREAM_RENDER_VULKAN_TEXTURE_HH
#define JETSTREAM_RENDER_VULKAN_TEXTURE_HH

#include "jetstream/render/base/texture.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class TextureImp<Device::Vulkan> : public Texture {
 public:
    explicit TextureImp(const Config& config);

    using Render::Texture::size;
    bool size(const Size2D<U64>& size);

    Result fill();
    Result fillRow(const U64& y, const U64& height);

    void* raw() {
        // return texture;
        return nullptr;
    }

 protected:
    Result create();
    Result destroy();

    // constexpr MTL::PixelFormat getPixelFormat() const {
    //     return pixelFormat;
    // }

    // constexpr MTL::Texture* getHandle() const {
    //     return texture;
    // }

    // static MTL::PixelFormat ConvertPixelFormat(const PixelFormat&, 
    //                                            const PixelType&);
    // static U64 GetPixelByteSize(const MTL::PixelFormat&);

 private:
    // MTL::Texture* texture = nullptr;
    // MTL::PixelFormat pixelFormat;

    friend class SurfaceImp<Device::Vulkan>;
    friend class ProgramImp<Device::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
