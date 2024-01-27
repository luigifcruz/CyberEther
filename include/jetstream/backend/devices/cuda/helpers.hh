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

#ifndef JST_NVRTC_CHECK
#define JST_NVRTC_CHECK(x, callback) { \
    nvrtcResult val = static_cast<nvrtcResult>((x)); \
    if (val != NVRTC_SUCCESS) { \
        const char* err = nvrtcGetErrorString(val); \
        (void)(err); \
        callback(); \
        return Result::ERROR; \
    } \
}
#endif  // JST_NVRTC_CHECK

#ifndef JST_NVRTC_CHECK_THROW
#define JST_NVRTC_CHECK_THROW(x, callback) { \
    nvrtcResult val = static_cast<nvrtcResult>((x)); \
    if (val != NVRTC_SUCCESS) { \
        const char* err = nvrtcGetErrorString(val, &err); \
        (void)(err); \
        callback(); \
        throw Result::ERROR; \
    } \
}
#endif  // JST_NVRTC_CHECK_THROW

namespace Jetstream::Backend {

}  // namespace Jetstream::Backend

#endif