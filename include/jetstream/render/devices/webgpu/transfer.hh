#ifndef JETSTREAM_RENDER_WEBGPU_TRANSFER_HH
#define JETSTREAM_RENDER_WEBGPU_TRANSFER_HH

#include <vector>

#include "jetstream/backend/base.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/base/transfer.hh"

namespace Jetstream::Render {

template<>
class JETSTREAM_API TransferImp<DeviceType::WebGPU> : public Transfer {
 public:
    Result encode(Batch& batch,
                  WGPUQueue queue,
                  WGPUCommandEncoder encoder);
    void destroy();

 private:
    static constexpr U64 MaximumPageSize = 64 * 1024 * 1024;
    static constexpr U64 RowAlignment = 256;

    struct StagingBuffer {
        WGPUBuffer buffer = nullptr;
        U64 capacity = 0;
    };

    std::vector<StagingBuffer> stagingBuffers;
    std::vector<U8> data;

    Result ensureCapacity(const U64& page,
                          const U64& required,
                          const U64& maximumBufferSize);
};

}  // namespace Jetstream::Render

#endif
