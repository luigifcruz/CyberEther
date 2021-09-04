#ifndef JETSTREAM_HELPERS_H
#define JETSTREAM_HELPERS_H

#ifndef JETSTREAM_CHECK_THROW
#define JETSTREAM_CHECK_THROW(result) { \
    if (result != Jetstream::Result::SUCCESS) { \
        std::cerr << "Jetstream encountered an exception at " \
            << __PRETTY_FUNCTION__ << " in line " <<  __LINE__<< " of file " \
            << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

#ifndef JETSTREAM_CHECK
#define JETSTREAM_CHECK(result) { \
    if (result != Jetstream::Result::SUCCESS) { \
        std::cerr << "Jetstream encountered an exception at " \
            << __PRETTY_FUNCTION__ << " in line " <<  __LINE__<< " of file " \
            << __FILE__ << "." << std::endl; \
        return result; \
    } \
}
#endif

#if __has_include("cuda_runtime.h") && defined JETSTREAM_CUDA_AVAILABLE
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(result) { \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA encountered an exception (" \
            << std::string(cudaGetErrorString(code)) << ") in " \
            << __PRETTY_FUNCTION__ << " in line " <<  __LINE__<< " of file " \
            << __FILE__ << "." << std::endl; \
        return Jetstream::Result::CUDA_ERROR; \
    } \
}
#endif

#ifndef CUDA_CHECK_THROW
#define CUDA_CHECK_THROW(result) { \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA encountered an exception (" \
            << std::string(cudaGetErrorString(code)) << ") in " \
            << __PRETTY_FUNCTION__ << " in line " <<  __LINE__<< " of file " \
            << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

#else
#ifndef CUDA_CHECK
#define CUDA_CHECK(result)
#endif
#ifndef CUDA_CHECK_THROW
#define CUDA_CHECK_THROW(result)
#endif
#endif

#if __has_include("nvtx3/nvToolsExt.h") && defined JETSTREAM_CUDA_AVAILABLE && defined JETSTREAM_DEBUG
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

#endif
