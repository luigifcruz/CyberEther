#ifndef JETSTREAM_RENDER_VULKAN_BUFFER_HH
#define JETSTREAM_RENDER_VULKAN_BUFFER_HH

#include "jetstream/render/base/buffer.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class BufferImp<DeviceType::Vulkan> : public Buffer {
 public:
    explicit BufferImp(const Config& config);

    Result create() override;
    Result destroy() override;

    using Render::Buffer::size;
    using Render::Buffer::byteSize;

    Result update() override;
    Result update(const U64& offset, const U64& size) override;

 protected:
    constexpr const VkBuffer& getHandle() const {
        return buffer;
    }

 private:
    VkBuffer buffer;
    VkDeviceMemory memory;

    friend class SurfaceImp<DeviceType::Vulkan>;
    friend class ProgramImp<DeviceType::Vulkan>;
    friend class VertexImp<DeviceType::Vulkan>;
    friend class KernelImp<DeviceType::Vulkan>;
    friend class TextureImp<DeviceType::Vulkan>;
    friend class DrawImp<DeviceType::Vulkan>;
};

}  // namespace Jetstream::Render

#endif
