#ifndef JETSTREAM_BENCHMARK_HH
#define JETSTREAM_BENCHMARK_HH

#include <iostream>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"

#include <nanobench.h>

namespace Jetstream {

class Benchmark {
 public:
    struct ResultEntry {
        std::string name;
        F64 ops_per_sec;
        F64 ms_per_op;
        F64 error;
    };

    typedef std::function<void(ankerl::nanobench::Bench& bench, std::string name)> BenchmarkFuncType;
    typedef std::map<std::string, std::vector<std::pair<std::string, BenchmarkFuncType>>> BenchmarkMapType;
    typedef std::map<std::string, std::vector<ResultEntry>> ResultMapType;

    static void Add(const std::string& module,
                    const std::string& device,
                    const std::string& type,
                    const BenchmarkFuncType& benchmark) {
        getInstance().add(module, device, type, benchmark);
    }

    static void Run(const std::string& outputType, std::ostream& out = std::cout) {
        getInstance().run(outputType, out);
    }

    static U64 TotalCount() {
        return getInstance().totalCount();
    }

    static U64 CurrentCount() {
        return getInstance().currentCount();
    }

    static void ResetResults() {
        getInstance().resetResults();
    }

    static const ResultMapType& GetResults() {
        return getInstance().getResults();
    }

 private:
    static Benchmark& getInstance();

    BenchmarkMapType benchmarks;
    ResultMapType results;

    U64 totalCount();
    U64 currentCount();
    void resetResults();
    const ResultMapType& getResults();

    void add(const std::string& module,
             const std::string& device,
             const std::string& type,
             const BenchmarkFuncType& benchmark);
    void run(const std::string& outputType, std::ostream& out);
};

}  // namespace Jetstream

#endif
