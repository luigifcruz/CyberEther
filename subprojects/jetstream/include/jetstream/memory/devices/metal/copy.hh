#ifndef JETSTREAM_MEMORY_METAL_COPY_HH
#define JETSTREAM_MEMORY_METAL_COPY_HH

#include "jetstream/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream::Memory {

template<typename DataType>
static Result Copy(Vector<Device::Metal, DataType>& dst,
                   const Vector<Device::Metal, DataType>& src) {
    if (dst.size() != src.size()) {
        JST_FATAL("Size mismatch between source and destination ({}, {}).",
                src.size(), dst.size());
        return Result::ASSERTION_ERROR;
    }

    memcpy(dst.data(), src.data(), src.size_bytes());

    return Memory::Copy(dst, src);
}

}  // namespace Jetstream::Memory

#endif
