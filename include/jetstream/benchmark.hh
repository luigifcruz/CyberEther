#ifndef JETSTREAM_BENCHMARK_HH
#define JETSTREAM_BENCHMARK_HH

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/memory/types.hh"
#include "jetstream/runtime.hh"
#include "jetstream/provider.hh"
#include "jetstream/parser.hh"
#include "jetstream/registry.hh"

namespace Jetstream {

class Module;

class JETSTREAM_API Benchmark {
 public:
    struct ResultEntry {
        std::string name;
        F64 opsPerSec;
        F64 msPerOp;
        F64 error;
    };

    struct InputSpec {
        std::string name;
        DataType dtype;
        Shape shape;
    };

    struct BenchmarkSpec {
        std::string variant;
        std::vector<InputSpec> inputs;
        Parser::Map config;
    };

    using BenchmarkSpecFunc = std::function<std::vector<BenchmarkSpec>()>;
    using ResultMapType = std::map<std::string, std::vector<ResultEntry>>;

    struct ModuleBenchmarkSpecs {
        std::string moduleType;
        BenchmarkSpecFunc specFunc;
    };

    static void RegisterModuleBenchmarkSpecs(const std::string& moduleType,
                                             BenchmarkSpecFunc specFunc);
    static void Run(const std::string& outputType, std::ostream& out = std::cout);
    static U64 TotalCount();
    static U64 CurrentCount();
    static void ResetResults();
    static const ResultMapType& GetResults();
    static const std::vector<ModuleBenchmarkSpecs>& ListModuleBenchmarkSpecs();

 private:
    struct Impl;
    static Impl& benchmark();
};

}  // namespace Jetstream

#define JST_BENCHMARK_INPUT(name, dtype, ...) \
    ::Jetstream::Benchmark::InputSpec{name, ::Jetstream::DataType::dtype, {__VA_ARGS__}}

#define JST_BENCHMARK_CONFIG(...) \
    [&]() { \
        ::Jetstream::Parser::Map m; \
        (__VA_ARGS__).serialize(m); \
        return m; \
    }()

#define JST_DETAIL_BENCHMARKS(module_type_str, id) \
    static std::vector<::Jetstream::Benchmark::BenchmarkSpec> \
    JST_DETAIL_CONCAT(__jst_benchmark_specs_, id)(); \
    namespace { \
    [[maybe_unused]] static const bool JST_DETAIL_CONCAT(__jst_register_benchmarks_, id) = []() { \
        ::Jetstream::Benchmark::RegisterModuleBenchmarkSpecs( \
            module_type_str, \
            &JST_DETAIL_CONCAT(__jst_benchmark_specs_, id) \
        ); \
        return true; \
    }(); \
    } \
    static std::vector<::Jetstream::Benchmark::BenchmarkSpec> \
    JST_DETAIL_CONCAT(__jst_benchmark_specs_, id)()

#define JST_BENCHMARKS(module_type_str) \
    JST_DETAIL_BENCHMARKS(module_type_str, __COUNTER__)

#endif  // JETSTREAM_BENCHMARK_HH
