#ifndef JETSTREAM_RENDER_METAL_TRANSFER_HH
#define JETSTREAM_RENDER_METAL_TRANSFER_HH

#include <array>

#include "jetstream/backend/base.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/base/transfer.hh"

namespace Jetstream::Render {

template<>
class JETSTREAM_API TransferImp<DeviceType::Metal> : public Transfer {
 public:
    Result encode(Batch& batch,
                  MTL::CommandBuffer* commandBuffer,
                  size_t frameIndex);
    void destroy();

 private:
    static constexpr size_t FramesInFlight = 3;

    struct Arena {
        MTL::Buffer* buffer = nullptr;
        U64 capacity = 0;
    };

    std::array<Arena, FramesInFlight> arenas{};

    Result ensureCapacity(Arena& arena, const U64& required);
    void destroyArena(Arena& arena);
};

}  // namespace Jetstream::Render

#endif
