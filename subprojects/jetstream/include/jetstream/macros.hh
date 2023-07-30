#ifndef JETSTREAM_MACROS_HH
#define JETSTREAM_MACROS_HH

#include "jetstream/memory/macros.hh"

#include "jetstream_config.hh"

#ifdef __EMSCRIPTEN__
#define JETSTREAM_STATIC
#endif

#ifndef JETSTREAM_API
#define JETSTREAM_API __attribute__((visibility("default")))
#endif  // JETSTREAM_API

#ifndef JETSTREAM_HIDDEN
#define JETSTREAM_HIDDEN __attribute__((visibility("hidden")))
#endif  // JETSTREAM_HIDDEN

#if __has_include("cuda_runtime.h") && defined JETSTREAM_CUDA_AVAILABLE
#include <cuda_runtime.h>

#ifndef JST_CUDA_CHECK_KERNEL
#define JST_CUDA_CHECK_KERNEL(callback) { \
    cudaError_t val; \
    if ((val = cudaPeekAtLastError()) != cudaSuccess) { \
        auto err = cudaGetErrorString(val); \
        return callback(); \
    } \
}
#endif  // JST_CUDA_CHECK_KERNEL

#ifndef JST_CUDA_CHECK
#define JST_CUDA_CHECK(x, callback) { \
    cudaError_t val = (x); \
    if (val != cudaSuccess) { \
        auto err = cudaGetErrorString(val); \
        callback(); \
        return Result::CUDA_ERROR; \
    } \
}
#endif  // JST_CUDA_CHECK

#endif

#ifndef JST_CUDA_CHECK_THROW
#define JST_CUDA_CHECK_THROW(x, callback) { \
    cudaError_t val = (x); \
    if (val != cudaSuccess) { \
        auto err = cudaGetErrorString(val); \
        callback(); \
        throw Result::CUDA_ERROR; \
    } \
}
#endif  // JST_CUDA_CHECK_THROW

#ifndef JST_CHECK
#define JST_CHECK(x) { \
    Result val = (x); \
    if (val != Result::SUCCESS) { \
        return val; \
    } \
}
#endif  // JST_CHECK

#ifndef JST_CHECK_THROW
#define JST_CHECK_THROW(x) { \
    Result val = (x); \
    if (val != Result::SUCCESS) { \
        printf("Function %s (%s@%d) throwed!\n", __func__, __FILE__, __LINE__); \
        throw val; \
    } \
}
#endif  // JST_CHECK_THROW

#ifndef JST_ASSERT
#define JST_ASSERT(x) { \
    bool val = (x); \
    if (val != true) { \
        return Result::ASSERTION_ERROR; \
    } \
}
#endif  // JST_ASSERT

#ifndef JST_ASSERT_THROW
#define JST_ASSERT_THROW(x) { \
    bool val = (x); \
    if (val != true) { \
        printf("Function %s (%s@%d) throwed!\n", __func__, __FILE__, __LINE__); \
        throw Result::ASSERTION_ERROR; \
    } \
}
#endif  // JST_ASSERT

#ifndef JST_CATCH
#define JST_CATCH(x, callback) { \
    try { \
        (void)(x); \
    } catch (const std::exception& e) { \
        return callback(); \
    } \
}
#endif  // JST_CATCH


#ifndef JST_SERDES
#define JST_SERDES(...) \
    Result serdes_function(const Parser::SerDesOp& op, Parser::RecordMap& data) const { (void)op; (void)data; __VA_ARGS__ return Result::SUCCESS; } \
    Result operator<<(Parser::RecordMap& data) const { return serdes_function(Parser::SerDesOp::Deserialize, data); } \
    Result operator>>(Parser::RecordMap& data) const { return serdes_function(Parser::SerDesOp::Serialize, data); }
#endif  // JST_SERDES

#ifndef JST_SERDES_VAL
#define JST_SERDES_VAL(fieldName, fieldVar) JST_CHECK(Parser::SerDes(data, fieldName, fieldVar, op));
#endif  // JST_SERDES_VAL

template <typename... Args>
struct CountArgs {
    static constexpr int value = sizeof...(Args);
};

#endif
