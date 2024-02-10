#ifndef JETSTREAM_MEMORY_METAL_COPY_HH
#define JETSTREAM_MEMORY_METAL_COPY_HH

#include "jetstream/memory/devices/metal/buffer.hh"
#include "jetstream/memory/devices/metal/copy.hh"
#include "jetstream/memory/devices/metal/tensor.hh"

namespace Jetstream::Memory {

template<typename T>
inline Result Copy(Tensor<Device::Metal, T>& dst, Tensor<Device::Metal, T>& src) {
    if ((dst.size() == src.size()) &&
        (dst.shape() == src.shape()) &&
        (dst.contiguous() && src.contiguous())) {
        void* dst_ptr = static_cast<void*>(dst.data()->contents());
        const void* src_ptr = static_cast<void*>(src.data()->contents());

        std::memcpy(dst_ptr, src_ptr, dst.size_bytes());

        if (!dst.device_native() && !src.device_native()) {
            dst.data()->didModifyRange(NS::Range(0, dst.size_bytes()));
        }
        return Result::SUCCESS;
    }

    JST_ERROR("[METAL:COPY] Copy not implemented for non-contiguous tensors.");
    return Result::ERROR;
}

}  // namespace Jetstream::Memory

#endif
