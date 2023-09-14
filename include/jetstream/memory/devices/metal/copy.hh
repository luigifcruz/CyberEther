#ifndef JETSTREAM_MEMORY_METAL_COPY_HH
#define JETSTREAM_MEMORY_METAL_COPY_HH

#include "jetstream/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream::Memory {

template<typename DataType, U64 Dimensions>
static Result Copy(Vector<Device::Metal, DataType, Dimensions>& dst,
                   const Vector<Device::Metal, DataType, Dimensions>& src) {
    if (dst.size() != src.size()) {
        JST_ERROR("Size mismatch between source and destination ({}, {}).",
                src.size(), dst.size());
        return Result::ERROR;
    }

    memcpy(dst.data(), src.data(), src.size_bytes());

    return Memory::Copy(dst, src);
}

}  // namespace Jetstream::Memory

#endif
