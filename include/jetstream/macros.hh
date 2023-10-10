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
        return Result::ERROR; \
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
        throw Result::ERROR; \
    } \
}
#endif  // JST_CUDA_CHECK_THROW

#ifndef JST_CHECK
#define JST_CHECK(...) { \
    Result val = (__VA_ARGS__); \
    if (val != Result::SUCCESS) { \
        return val; \
    } \
}
#endif  // JST_CHECK

#ifndef JST_CHECK_THROW
#define JST_CHECK_THROW(...) { \
    Result val = (__VA_ARGS__); \
    if (val != Result::SUCCESS) { \
        printf("Function %s (%s@%d) throwed!\n", __func__, __FILE__, __LINE__); \
        throw val; \
    } \
}
#endif  // JST_CHECK_THROW

#ifndef JST_ASSERT
#define JST_ASSERT(...) { \
    if (!(__VA_ARGS__)) { \
        return Result::ERROR; \
    } \
}
#endif  // JST_ASSERT

#ifndef JST_ASSERT_THROW
#define JST_ASSERT_THROW(...) { \
    if (!(__VA_ARGS__)) { \
        printf("Function %s (%s@%d) throwed!\n", __func__, __FILE__, __LINE__); \
        throw Result::ERROR; \
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

// Module construction

#ifndef JST_DEFINE_IO
#define JST_DEFINE_IO() protected: Config config; Input input; Output output; friend Instance;
#endif  // JST_DEFINE_IO

// Struct serialize deserialize

#ifndef JST_SERDES
#define JST_SERDES(...) \
    Result operator<<(Parser::RecordMap& data) { const auto op = Parser::SerDesOp::Deserialize; (void)op; (void)data; __VA_ARGS__ return Result::SUCCESS; } \
    Result operator>>(Parser::RecordMap& data) { const auto op = Parser::SerDesOp::Serialize;   (void)op; (void)data; __VA_ARGS__ return Result::SUCCESS; }
#endif  // JST_SERDES

#ifndef JST_SERDES_VAL
#define JST_SERDES_VAL(fieldName, fieldVar) JST_CHECK(Parser::SerDes(data, fieldName, fieldVar, op));
#endif  // JST_SERDES_VAL

// Module Buffer Initialization

#ifndef JST_INIT
#define JST_INIT(...) \
Result res = Result::SUCCESS; \
__VA_ARGS__ \
JST_CHECK(res);
#endif  // JST_INIT

#ifndef JST_INIT_INPUT
#define JST_INIT_INPUT(fieldName, fieldVar) res |= this->initInput(fieldName, fieldVar);
#endif  // JST_INIT_INPUT

#ifndef JST_INIT_OUTPUT
#define JST_INIT_OUTPUT(fieldName, fieldVar, fieldShape) res |= this->initOutput(fieldName, fieldVar, fieldShape, res);
#endif  // JST_INIT_OUTPUT

#ifndef JST_VOID_OUTPUT
#define JST_VOID_OUTPUT(fieldVar) JST_CHECK(this->voidOutput(fieldVar));
#endif  // JST_VOID_OUTPUT

// Miscellaneous

#include <thread>

#ifndef JST_DISPATCH_ASYNC
#define JST_DISPATCH_ASYNC(...) std::thread(__VA_ARGS__).detach();
#endif  // JST_DISPATCH_ASYNC

template <typename... Args>
struct CountArgs {
    static constexpr int value = sizeof...(Args);
};

#endif
