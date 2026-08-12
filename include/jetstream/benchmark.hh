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

namespace Jetstream {

class JETSTREAM_API Benchmark {
 public:
    struct Input {
        std::string name;
        DataType dtype;
        Shape shape;
    };

    struct Case {
        std::string variant;
        std::vector<Input> inputs;
        Parser::Map config;
    };

    struct Measurement {
        std::string name;
        F64 opsPerSec;
        F64 msPerOp;
        F64 error;
    };

    using ResultMapType = std::map<std::string, std::vector<Measurement>>;

    static void Run(const std::string& outputType, std::ostream& out = std::cout);
    static void Run(const std::string& outputType,
                    const std::string& blockType,
                    std::ostream& out = std::cout);
    static U64 TotalCount(const std::string& blockType = "");
    static U64 CurrentCount();
    static void ResetResults();
    static const ResultMapType& GetResults();

 private:
    struct Impl;
    static Impl& benchmark();
};

}  // namespace Jetstream

#define JST_BENCHMARK_INPUT(name, dtype, ...) \
    ::Jetstream::Benchmark::Input{name, ::Jetstream::DataType::dtype, {__VA_ARGS__}}

#define JST_BENCHMARK_CONFIG(...) \
    [&]() { \
        ::Jetstream::Parser::Map m; \
        (__VA_ARGS__).serialize(m); \
        return m; \
    }()

#endif  // JETSTREAM_BENCHMARK_HH
