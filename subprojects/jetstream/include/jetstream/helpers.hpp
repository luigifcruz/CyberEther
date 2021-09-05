#ifndef JETSTREAM_HELPERS_H
#define JETSTREAM_HELPERS_H

#include "jetstream_config.hpp"
#include <iostream>

#ifndef JST_CHECK_THROW
#define JST_CHECK_THROW(result) { \
    if (result != Jetstream::Result::SUCCESS) { \
        std::cerr << "Jetstream encountered an exception at " \
            << __PRETTY_FUNCTION__ << " in line " <<  __LINE__<< " of file " \
            << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

#ifndef JST_CHECK
#define JST_CHECK(result) { \
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
void jst_cuda_print_error(cudaError_t, const char*, int, const char*);

#ifndef JST_CUDA_CHECK
#define JST_CUDA_CHECK(result) { \
    if (result != cudaSuccess) { \
        jst_cuda_print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        return Jetstream::Result::ERROR; \
    } \
}
#endif

#ifndef JST_CUDA_CHECK_THROW
#define JST_CUDA_CHECK_THROW(result) { \
    if (result != cudaSuccess) { \
        jst_cuda_print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        throw result; \
    } \
}
#endif

#else
#ifndef JST_CUDA_CHECK
#define JST_CUDA_CHECK(result)
#endif
#ifndef JST_CUDA_CHECK_THROW
#define JST_CUDA_CHECK_THROW(result)
#endif
#endif

#if __has_include("nvtx3/nvToolsExt.h") && defined JETSTREAM_CUDA_AVAILABLE && defined JETSTREAM_DEBUG
#include <nvtx3/nvToolsExt.h>

#ifndef DEBUG_PUSH
inline void DEBUG_PUSH(std::string name) { nvtxRangePushA(name.c_str()); }
inline void DEBUG_PUSH(const char* name) { nvtxRangePushA(name); }
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
