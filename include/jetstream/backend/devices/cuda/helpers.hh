#ifndef JETSTREAM_BACKEND_DEVICE_CUDA_HELPERS_HH
#define JETSTREAM_BACKEND_DEVICE_CUDA_HELPERS_HH

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef JST_CUDA_CHECK
#define JST_CUDA_CHECK(x, callback) { \
    CUresult val = static_cast<CUresult>((x)); \
    if (val != CUDA_SUCCESS) { \
        const char* err = nullptr; \
        cuGetErrorString(val, &err); \
        callback(); \
        return Result::ERROR; \
    } \
}
#endif  // JST_CUDA_CHECK

#ifndef JST_CUDA_CHECK_THROW
#define JST_CUDA_CHECK_THROW(x, callback) { \
    CUresult val = static_cast<CUresult>((x)); \
    if (val != CUDA_SUCCESS) { \
        const char* err = nullptr; \
        cuGetErrorString(val, &err); \
        callback(); \
        throw Result::ERROR; \
    } \
}
#endif  // JST_CUDA_CHECK_THROW

namespace Jetstream::Backend {

}  // namespace Jetstream::Backend

#endif