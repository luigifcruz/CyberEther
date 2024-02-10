#ifndef JETSTREAM_MEMORY_CPU_COPY_HH
#define JETSTREAM_MEMORY_CPU_COPY_HH

#include "jetstream/memory/devices/cpu/buffer.hh"
#include "jetstream/memory/devices/cpu/copy.hh"
#include "jetstream/memory/devices/cpu/tensor.hh"

namespace Jetstream::Memory {

template<typename T>
inline Result Copy(Tensor<Device::CPU, T>& dst, const Tensor<Device::CPU, T>& src) {
    if ((dst.size() == src.size()) &&
        (dst.shape() == src.shape()) &&
        (dst.contiguous() && src.contiguous())) {
        std::memcpy(dst.data(), src.data(), dst.size_bytes());
        return Result::SUCCESS;
    }

    JST_ERROR("[CPU:COPY] Copy not implemented for non-contiguous tensors.");
    return Result::ERROR;
}

}  // namespace Jetstream::Memory

#endif
