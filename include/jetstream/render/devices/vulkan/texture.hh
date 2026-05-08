#ifndef JETSTREAM_RENDER_VULKAN_TEXTURE_HH
#define JETSTREAM_RENDER_VULKAN_TEXTURE_HH

#include "jetstream/render/base/texture.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class TextureImp<DeviceType::Vulkan> : public Texture {
 public:
    explicit TextureImp(const Config& config);

    Result create() override;
    Result destroy() override;

    using Render::Texture::size;
    bool size(const Extent2D<U64>& size) override;

    Result fill() override;
    Result fillRow(const U64& y, const U64& height) override;

    uint64_t raw() const override {
        return descriptorSet ? (uint64_t)(void*)descriptorSet : 0;
    }

 protected:
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
    VkImage texture = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkFormat pixelFormat;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkExtent2D extent = {};

    friend class SurfaceImp<DeviceType::Vulkan>;
    friend class ProgramImp<DeviceType::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
