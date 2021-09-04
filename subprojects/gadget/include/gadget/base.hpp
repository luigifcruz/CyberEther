#ifndef GADGET_H
#define GADGET_H

#include <iostream>
#include <complex>

#if __has_include("cuda_runtime.h")
#define GADGET_HAS_CUDA
#endif

#if defined GADGET_HAS_CUDA
#include <cuda_runtime.h>

#ifndef GT_CUDA_CHECK_THROW
#define GT_CUDA_CHECK_THROW(result) { \
    if (result != cudaSuccess) { \
        cuda_print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        throw result; \
    } \
}
#endif

#ifndef GT_CUDA_CHECK
#define GT_CUDA_CHECK(result) { \
    if (result != cudaSuccess) { \
        cuda_print_error(result, __PRETTY_FUNCTION__, __LINE__, __FILE__); \
        return Gadget::Result::CUDA_ERROR; \
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

#endif
