#ifndef JETSTREAM_TYPE_H
#define JETSTREAM_TYPE_H

#include <complex>
#include <future>
#include <iostream>
#include <vector>

#include "tools/magic_enum.hpp"
#include "jetstream_config.hpp"
#include "tools/span.hpp"

#ifdef CUDA_DEBUG
#include <nvtx3/nvToolsExt.h>

#ifndef DEBUG_PUSH
#define DEBUG_PUSH(name) { nvtxRangePushA(name); }
#endif
#ifndef DEBUG_POP
#define DEBUG_POP() { nvtxRangePop(); }
#endif

#else

#ifndef DEBUG_PUSH
#define DEBUG_PUSH(name)
#endif
#ifndef DEBUG_POP
#define DEBUG_POP()
#endif

#endif

#ifndef JETSTREAM_ASSERT_SUCCESS
#define JETSTREAM_ASSERT_SUCCESS(result) { \
    if (result != Jetstream::Result::SUCCESS) { \
        std::cerr << "Jetstream encountered an exception (" <<  magic_enum::enum_name(result) << ") in " \
            << __PRETTY_FUNCTION__ << " in line " << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

namespace Jetstream {

enum Result {
    SUCCESS = 0,
    ERROR = 1,
    UNKNOWN = 2,
    TIMEOUT,
    ERROR_DATA_DEPENDECY,
    ERROR_FUTURE_INVALID,
};

enum class Launch : uint8_t {
    ASYNC   = 1,
    SYNC    = 2,
};

enum class Locale : uint8_t {
    NONE    = 0 << 0,
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
};

template<typename T>
struct Data {
    Locale location;
    T buf;
};

class Module;
typedef std::vector<std::shared_ptr<Module>> Graph;
typedef struct { Launch launch; Graph deps; } Policy;

} // namespace Jetstream

#endif
