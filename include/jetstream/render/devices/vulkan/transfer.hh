#ifndef JETSTREAM_RENDER_VULKAN_TRANSFER_HH
#define JETSTREAM_RENDER_VULKAN_TRANSFER_HH

#include <vector>

#include "jetstream/backend/base.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/base/transfer.hh"

namespace Jetstream::Render {

template<>
class JETSTREAM_API TransferImp<DeviceType::Vulkan> : public Transfer {
 public:
    void create(size_t framesInFlight);
    void destroy();
    Result encode(Batch& batch,
                  VkCommandBuffer commandBuffer,
                  size_t frameIndex);
    void commit(const Batch& batch);

 private:
    struct Arena {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        U8* mapped = nullptr;
        U64 capacity = 0;
    };

    std::vector<Arena> arenas;

    Result ensureCapacity(Arena& arena, const U64& required);
    void destroyArena(Arena& arena);
};

}  // namespace Jetstream::Render

#endif
