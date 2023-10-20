#ifndef JETSTREAM_RENDER_VULKAN_TEXTURE_HH
#define JETSTREAM_RENDER_VULKAN_TEXTURE_HH

#include "jetstream/render/tools/imgui_impl_vulkan.h"

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
        return descriptorSet;
    }

    Result create();
    Result destroy();

    constexpr const VkFormat& getPixelFormat() const {
        return pixelFormat;
    }

    constexpr const VkImage& getHandle() const {
        return texture;
    }

    constexpr const VkImageView& getViewHandle() const {
        return imageView;
    }

    constexpr const VkSampler& getSamplerHandler() const {
        return sampler;
    }

    constexpr const VkExtent2D& getExtent() const {
        return extent;
    }

    static VkFormat ConvertPixelFormat(const PixelFormat&, 
                                       const PixelType&);
    static U64 GetPixelByteSize(const VkFormat&);

 private:
    VkImage texture;
    VkImageView imageView;
    VkSampler sampler;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    VkFormat pixelFormat;
    VkDeviceMemory memory;
    VkExtent2D extent;
};

}  // namespace Jetstream::Render

#endif
