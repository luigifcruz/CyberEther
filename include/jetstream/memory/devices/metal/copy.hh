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
        T* dst_ptr = static_cast<T*>(dst.data()->contents());
        const T* src_ptr = static_cast<T*>(src.data()->contents());

        std::memcpy(dst_ptr + dst.offset(), src_ptr + src.offset(), dst.size_bytes());

        if (!dst.device_native() && !src.device_native()) {
            dst.data()->didModifyRange(NS::Range(dst.offset_bytes(), dst.offset_bytes() + dst.size_bytes()));
        }
        return Result::SUCCESS;
    }

    JST_ERROR("[METAL:COPY] Copy not implemented for non-contiguous tensors.");
    return Result::ERROR;
}

template<typename T>
inline Result Copy(Tensor<Device::CPU, T>& dst, Tensor<Device::Metal, T>& src) {
    return Copy(dst.metal(), src);
}

}  // namespace Jetstream::Memory

#endif
