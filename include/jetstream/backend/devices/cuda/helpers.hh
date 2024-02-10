#ifndef JETSTREAM_BACKEND_DEVICE_CUDA_HELPERS_HH
#define JETSTREAM_BACKEND_DEVICE_CUDA_HELPERS_HH

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cufft.h>

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
        const char* err = nvrtcGetErrorString(val); \
        (void)(err); \
        callback(); \
        throw Result::ERROR; \
    } \
}
#endif  // JST_NVRTC_CHECK_THROW

#ifndef JST_CUFFT_CHECK
#define JST_CUFFT_CHECK(x, callback) { \
    cufftResult val = static_cast<cufftResult>((x)); \
    if (val != CUFFT_SUCCESS) { \
        const char* err = cufftGetErrorString(val); \
        (void)(err); \
        callback(); \
        throw Result::ERROR; \
    } \
}
#endif  // JST_CUFFT_CHECK

#ifndef JST_CUFFT_CHECK_THROW
#define JST_CUFFT_CHECK_THROW(x, callback) { \
    cufftResult val = static_cast<cufftResult>((x)); \
    if (val != CUFFT_SUCCESS) { \
        const char* err = cufftGetErrorString(val); \
        (void)(err); \
        callback(); \
        throw Result::ERROR; \
    } \
}
#endif  // JST_CUFFT_CHECK_THROW

namespace Jetstream {

inline const char* cufftGetErrorString(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS: The cuFFT operation was successful.";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN: cuFFT was passed an invalid plan handle.";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED: cuFFT failed to allocate GPU or CPU memory.";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE: No longer used.";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE: User specified an invalid pointer or parameter.";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR: Driver or internal cuFFT library error.";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED: Failed to execute an FFT on the GPU.";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED: The cuFFT library failed to initialize.";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE: User specified an invalid transform size.";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA: No longer used.";
        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST: Missing parameters in call.";
        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE: Execution of a plan was on different GPU than plan creation.";
        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR: Internal plan database error.";
        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE: No workspace has been provided prior to plan execution.";
        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED: Function does not implement functionality for parameters given.";
        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR: Used in previous versions.";
        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED: Operation is not supported for parameters given.";
        default:
            return "Unknown error";
    }
}       

}  // namespace Jetstream

#endif