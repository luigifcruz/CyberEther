#ifndef JETSTREAM_MACROS_HH
#define JETSTREAM_MACROS_HH

#include <cstdio>
#include <exception>
#include <thread>

#include "jetstream/config.hh"

//
// Core macros.
//

#ifndef JETSTREAM_API
// TODO: Add support for DLL export if needed.
#if defined(JST_IS_STATIC) || defined(JST_OS_WINDOWS)
#define JETSTREAM_API
#elif defined(__GNUC__) || defined(__clang__)
#define JETSTREAM_API __attribute__((visibility("default")))
#else
#define JETSTREAM_API
#endif
#endif  // JETSTREAM_API

#ifndef JETSTREAM_HIDDEN
#if defined(JST_IS_STATIC) || defined(JST_OS_WINDOWS)
#define JETSTREAM_HIDDEN
#elif defined(__GNUC__) || defined(__clang__)
#define JETSTREAM_HIDDEN __attribute__((visibility("hidden")))
#else
#define JETSTREAM_HIDDEN
#endif
#endif  // JETSTREAM_HIDDEN

//
// Check macros.
//

#ifndef JST_CHECK
#define JST_CHECK(...) { \
    Result val = (__VA_ARGS__); \
    if (val != Result::SUCCESS && val != Result::RELOAD) { \
        return val; \
    } \
}
#endif  // JST_CHECK

#ifndef JST_CHECK_ALLOW
#define JST_CHECK_ALLOW(expr, ...) { \
    Result val = (expr); \
    if (val != Result::SUCCESS && val != Result::RELOAD \
        FOR_EACH(JST_CHECK_ALLOW_CMP, __VA_ARGS__)) { \
        return val; \
    } \
}
#define JST_CHECK_ALLOW_CMP(x) && val != x
#endif  // JST_CHECK_ALLOW

#ifndef JST_CHECK_THROW
#define JST_CHECK_THROW(...) { \
    Result val = (__VA_ARGS__); \
    if (val != Result::SUCCESS && val != Result::RELOAD) { \
        printf("Function %s (%s@%d) throwed!\n", __func__, __FILE__, __LINE__); \
        throw val; \
    } \
}
#endif  // JST_CHECK_THROW

#ifndef JST_ASSERT
#define JST_ASSERT(condition, ...) { \
    if (!(condition)) { \
        JST_ERROR(__VA_ARGS__); \
        return Result::ERROR; \
    } \
}
#endif  // JST_ASSERT

#ifndef JST_ASSERT_THROW
#define JST_ASSERT_THROW(condition, ...) { \
    if (!(condition)) { \
        JST_ERROR(__VA_ARGS__); \
        throw Result::ERROR; \
    } \
}
#endif  // JST_ASSERT_THROW

#ifndef JST_CATCH
#define JST_CATCH(x, callback) { \
    try { \
        (void)(x); \
    } catch (const std::exception& e) { \
        return callback(); \
    } \
}
#endif  // JST_CATCH

//
// Variardic for_each mechanism.
// Source: https://www.scs.stanford.edu/~dm/blog/va-opt.html
//

#ifndef FOR_EACH
#define PARENS ()

#define EXPAND(...) EXPAND4(EXPAND4(EXPAND4(EXPAND4(__VA_ARGS__))))
#define EXPAND4(...) EXPAND3(EXPAND3(EXPAND3(EXPAND3(__VA_ARGS__))))
#define EXPAND3(...) EXPAND2(EXPAND2(EXPAND2(EXPAND2(__VA_ARGS__))))
#define EXPAND2(...) EXPAND1(EXPAND1(EXPAND1(EXPAND1(__VA_ARGS__))))
#define EXPAND1(...) __VA_ARGS__

#define FOR_EACH(macro, ...)                                    \
  __VA_OPT__(EXPAND(FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define FOR_EACH_HELPER(macro, a1, ...)                         \
  macro(a1)                                                     \
  __VA_OPT__(FOR_EACH_AGAIN PARENS (macro, __VA_ARGS__))
#define FOR_EACH_AGAIN() FOR_EACH_HELPER
#endif  // FOR_EACH

//
// Miscellaneous macros.
//

#ifndef JST_DISPATCH_ASYNC
#define JST_DISPATCH_ASYNC(...) std::thread(__VA_ARGS__).detach();
#endif  // JST_DISPATCH_ASYNC

#endif
