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
    WGPUBuffer buffer = nullptr;
    U64 capacity = 0;
    std::vector<U8> data;

    Result ensureCapacity(const U64& required);
};

}  // namespace Jetstream::Render

#endif
