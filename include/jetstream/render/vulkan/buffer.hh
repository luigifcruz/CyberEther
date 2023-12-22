#ifndef JETSTREAM_RENDER_VULKAN_BUFFER_HH
#define JETSTREAM_RENDER_VULKAN_BUFFER_HH

#include "jetstream/render/base/buffer.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class BufferImp<Device::Vulkan> : public Buffer {
 public:
    explicit BufferImp(const Config& config);

    Result create();
    Result destroy();

    using Render::Buffer::size;
    using Render::Buffer::byteSize;

    Result update();
    Result update(const U64& offset, const U64& size);

 protected:
    constexpr const VkBuffer& getHandle() const {
        return buffer;
    }

    constexpr const VkDescriptorType& getDescriptorType() const {
        return descriptorType;
    }

 private:
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDescriptorType descriptorType;

    friend class SurfaceImp<Device::Vulkan>;
    friend class ProgramImp<Device::Vulkan>;
    friend class VertexImp<Device::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
