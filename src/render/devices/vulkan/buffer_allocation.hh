#ifndef JETSTREAM_RENDER_VULKAN_BUFFER_ALLOCATION_HH
#define JETSTREAM_RENDER_VULKAN_BUFFER_ALLOCATION_HH

#include <vulkan/vulkan.h>

#include "jetstream/types.hh"

namespace Jetstream::Render::Detail {

template<typename Create, typename Allocate, typename Bind, typename Cleanup>
Result AllocateBuffer(VkBuffer& buffer, VkDeviceMemory& memory,
                      Create&& create, Allocate&& allocate, Bind&& bind, Cleanup&& cleanup) {
    const auto rollback = [&]() {
        cleanup(buffer, memory);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    };
    Result result;
    try {
        VkBuffer candidateBuffer = VK_NULL_HANDLE;
        result = create(candidateBuffer);
        if (result == Result::SUCCESS || result == Result::RELOAD) {
            buffer = candidateBuffer;
            VkDeviceMemory candidateMemory = VK_NULL_HANDLE;
            result = allocate(buffer, candidateMemory);
            if (result == Result::SUCCESS || result == Result::RELOAD) {
                memory = candidateMemory;
                result = bind(buffer, memory);
            }
        }
    } catch (...) {
        rollback();
        throw;
    }
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        rollback();
    }
    return result;
}

}  // namespace Jetstream::Render::Detail

#endif
