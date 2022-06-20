#ifndef JETSTREAM_MEMORY_CPU_COPY_HH
#define JETSTREAM_MEMORY_CPU_COPY_HH

#include "jetstream/types.hh"
#include "jetstream/memory/vector.hh"

namespace Jetstream::Memory {

template<typename T>
static Result Copy(Vector<Device::CPU, T>& dst,
                   const Vector<Device::CPU, T>& src) {
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
