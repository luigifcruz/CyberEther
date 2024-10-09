#ifndef JETSTREAM_MACROS_HH
#define JETSTREAM_MACROS_HH

#include "jetstream/memory/macros.hh"

#include "jetstream/config.hh"

//
// Core macros.
//

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
#define JST_DEFINE_IO() \
    public: \
        void init_benchmark_mode(const Config& _c, const Input& _i) { \
            config = _c; \
            input = _i; \
            under_benchmark = true; \
        }; \
    protected: \
        Config config; \
        Input input; \
        Output output; \
        bool under_benchmark = false; \
        friend Instance;
#endif  // JST_DEFINE_IO

#ifndef JST_INIT_IO
#define JST_INIT_IO() \
    if (!under_benchmark) { \
        JST_CHECK(output.init(locale())); \
        JST_CHECK(input.init(taint())); \
    }
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
// Variardic concatenation mechanism.
//

#ifndef JST_UNIQUE
#define JST_UNIQUE_CONCAT_HELPER(A, B) A##B
#define JST_UNIQUE_CONCAT(A, B) JST_UNIQUE_CONCAT_HELPER(A, B)
#define JST_UNIQUE(A) JST_UNIQUE_CONCAT(A, __COUNTER__)
#endif  // JST_UNIQUE

//
// Benchmark macros.
//

#ifndef COMMA
#define COMMA ,
#endif  // COMMA

#ifndef JST_BENCHMARK_RUN
#define JST_BENCHMARK_RUN(TestName, Config, Input, ...) \
    { \
        auto graph = NewGraph(D); \
        auto module = std::make_shared<Module<D, __VA_ARGS__>>(); \
        module->init_benchmark_mode(Config, Input); \
        module->create(); \
        graph->setModule(module, {}, {}); \
        graph->create(); \
        bench.run(name + TestName, [&] { \
            std::unordered_set<U64> yielded; \
            graph->compute(yielded); \
        }); \
        graph->destroy(); \
        module->destroy(); \
    }
#endif  // JST_BENCHMARK_RUN

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
    JST_CHECK(Module::InitInput(var, taint));
#endif  // JST_INIT_INPUT_ACTION

#ifndef JST_SERDES_INPUT
#define JST_SERDES_INPUT(...) \
    JST_SERDES(__VA_ARGS__) \
    Result init(const Taint& taint) { \
        (void)taint; \
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
        (void)locale; \
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
template <> struct is_specialized<Class<Device:: DeviceType, __VA_ARGS__>> { static constexpr bool value = true; };

#define JST_INSTANTIATION(Class, DeviceType, ...) \
template class Class<Device:: DeviceType, __VA_ARGS__>;

#define JST_BENCHMARK(Class, DeviceType, ...) \
    static bool JST_UNIQUE_CONCAT(Class, JST_UNIQUE(_Benchmark_)) __attribute__((used)) = []() -> bool { \
        Jetstream::Benchmark::Add(#Class, #DeviceType, #__VA_ARGS__, benchmark<Class, Device:: DeviceType, __VA_ARGS__>); \
        return true; \
    }();

//
// Enum serialization macros.
//

#ifndef JST_NAME
#define JST_NAME(x) #x,
#endif  // JST_NAME

#ifndef JST_SERDES_ENUM
#define JST_SERDES_ENUM(EnumName, ...) \
class EnumName : public Parser::Adapter { \
 public: \
    enum Value : uint8_t { __VA_ARGS__ }; \
    EnumName() = default; \
    EnumName(Value val) : value(val) {} \
    operator int() const { return value; } \
    bool operator==(const Value& other) const { return value == other; } \
    bool operator!=(const Value& other) const { return value != other; } \
    static const std::map<Value, std::string>& rmap() { \
        static const std::map<Value, std::string> rmap = EnumName::CreateReverseMap(); \
        return rmap; \
    } \
    static const std::map<std::string, Value>& map() { \
        static const std::map<std::string, Value> map = EnumName::CreateMap(); \
        return map; \
    } \
    const std::string& string() const { \
        return rmap().at(value); \
    } \
    Result serialize(std::any& var) const override { \
        var = std::any(string()); \
        return Result::SUCCESS; \
    } \
    Result deserialize(const std::any& var) override { \
        const auto& str = std::any_cast<std::string>(var); \
        value = map().at(str); \
        return Result::SUCCESS; \
    } \
    friend std::ostream& operator<<(std::ostream& os, const EnumName& m) { \
        return os << m.string(); \
    } \
 private: \
    Value value; \
    static std::map<Value, std::string> CreateReverseMap() { \
        std::map<Value, std::string> rmap; \
        const char* names[] = {FOR_EACH(JST_NAME, __VA_ARGS__) nullptr}; \
        Value vals[] = {__VA_ARGS__}; \
        for (int i = 0; names[i]; i++) { \
            rmap[vals[i]] = names[i]; \
        } \
        return rmap; \
    } \
    static std::map<std::string, Value> CreateMap() { \
        std::map<std::string, Value> map; \
        const char* names[] = {FOR_EACH(JST_NAME, __VA_ARGS__) nullptr}; \
        Value vals[] = {__VA_ARGS__}; \
        for (int i = 0; names[i]; i++) { \
            map[names[i]] = vals[i]; \
        } \
        return map; \
    } \
};
#endif  // JST_SERDES_ENUM

//
// Block specialization macros.
//

#define JST_BLOCK_ENABLE(BLOCK, ...) \
namespace Jetstream { \
    template <Device D, typename IT, typename OT> \
    struct is_specialized<Blocks::BLOCK<D, IT, OT>> { \
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
