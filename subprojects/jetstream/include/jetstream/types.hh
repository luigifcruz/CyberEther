#ifndef JETSTREAM_TYPE_HH
#define JETSTREAM_TYPE_HH

#include <any>
#include <span>
#include <vector>
#include <complex>
#include <typeindex>
#include <unordered_map>

#include "jetstream/memory/types.hh"

namespace Jetstream {

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    SKIP = 2,
    RECREATE = 3,
    CUDA_ERROR,
    VULKAN_ERROR,
    ASSERTION_ERROR,
    ERROR_TIMEOUT,
    ERROR_BEYOND_CAPACITY,
};

enum class Direction : int8_t {
    Forward = 0,
    Backward = 1,
};

}  // namespace Jetstream

#endif
