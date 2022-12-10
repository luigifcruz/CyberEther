#ifndef JETSTREAM_TYPE_HH
#define JETSTREAM_TYPE_HH

#include <map>
#include <span>
#include <vector>
#include <complex>
#include <typeindex>

#include "jetstream/memory/types.hh"

namespace Jetstream {

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
    ERROR_TIMEOUT,
    ERROR_BEYOND_CAPACITY,
};

enum class Direction : int8_t {
    Forward = 1,
    Backward = -1,
};

}  // namespace Jetstream

#endif
