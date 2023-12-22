#ifndef JETSTREAM_MACROS_HH
#define JETSTREAM_MACROS_HH

#include "jetstream/memory/macros.hh"

#include "jetstream_config.hh"

//
// Core macros.
// 

#ifdef __EMSCRIPTEN__
#define JETSTREAM_STATIC
#endif

#ifndef JETSTREAM_API
#define JETSTREAM_API __attribute__((visibility("default")))
#endif  // JETSTREAM_API

#ifndef JETSTREAM_HIDDEN
#define JETSTREAM_HIDDEN __attribute__((visibility("hidden")))
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

//
// Module construction.
//

#ifndef JST_DEFINE_IO
#define JST_DEFINE_IO() protected: Config config; Input input; Output output; friend Instance;
#endif  // JST_DEFINE_IO

#ifndef JST_INIT_IO
#define JST_INIT_IO() \
    JST_CHECK(output.init(locale())); \
    JST_CHECK(input.init());
#endif  // JST_INIT_IO

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
// Struct serialize/deserialize methods.
//

#ifndef JST_SERDES_ACTION
#define JST_SERDES_ACTION(var) \
    JST_CHECK(Parser::SerDes(data, #var, var, op));
#endif  // JST_SERDES_ACTION

#ifndef JST_SERDES
#define JST_SERDES(...) \
    Result operator<<(Parser::RecordMap& data) { \
        const auto op = Parser::SerDesOp::Deserialize; \
        (void)op; (void)data; \
        FOR_EACH(JST_SERDES_ACTION, __VA_ARGS__) \
        return Result::SUCCESS; \
    } \
    Result operator>>(Parser::RecordMap& data) { \
        const auto op = Parser::SerDesOp::Serialize; \
        (void)op; (void)data; \
        FOR_EACH(JST_SERDES_ACTION, __VA_ARGS__) \
        return Result::SUCCESS; \
    }
#endif  // JST_SERDES

#ifndef JST_INIT_INPUT_ACTION
#define JST_INIT_INPUT_ACTION(var) \
    JST_CHECK(Module::InitInput(var));
#endif  // JST_INIT_INPUT_ACTION

#ifndef JST_SERDES_INPUT
#define JST_SERDES_INPUT(...) \
    JST_SERDES(__VA_ARGS__) \
    Result init() { \
        FOR_EACH(JST_INIT_INPUT_ACTION, __VA_ARGS__) \
        return Result::SUCCESS; \
    }
#endif  // JST_SERDES_INPUT

#ifndef JST_INIT_OUTPUT_ACTION
#define JST_INIT_OUTPUT_ACTION(var) \
    JST_CHECK(Module::InitOutput(#var, var, locale));
#endif  // JST_INIT_OUTPUT_ACTION

#ifndef JST_SERDES_OUTPUT
#define JST_SERDES_OUTPUT(...) \
    JST_SERDES(__VA_ARGS__) \
    Result init(const Locale& locale) { \
        FOR_EACH(JST_INIT_OUTPUT_ACTION, __VA_ARGS__) \
        return Result::SUCCESS; \
    }
#endif  // JST_SERDES_OUTPUT

//
// Module specialization macros.
//

namespace Jetstream {

template <typename>
struct is_specialized {
    static constexpr bool value = false;
};

}  // namespace Jetstream

#define JST_SPECIALIZATION(Class, DeviceType, ...) \
template <> struct is_specialized<Class<DeviceType, __VA_ARGS__>> { static constexpr bool value = true; };

#define JST_INSTANTIATION(Class, DeviceType, ...) \
template class Class<DeviceType, __VA_ARGS__>;

//
// Block specialization macros.
//

#define JST_BLOCK_ENABLE(BLOCK, ...) \
namespace Jetstream { \
    template <Device D, typename IT, typename OT> \
    struct Jetstream::is_specialized<Blocks::BLOCK<D, IT, OT>> { \
        static constexpr bool value = (__VA_ARGS__); \
    }; \
}

//
// Miscellaneous macros.
//

#include <thread>

#ifndef JST_DISPATCH_ASYNC
#define JST_DISPATCH_ASYNC(...) std::thread(__VA_ARGS__).detach();
#endif  // JST_DISPATCH_ASYNC

template <typename... Args>
struct CountArgs {
    static constexpr int value = sizeof...(Args);
};

#endif
