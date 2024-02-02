#include <chrono>

#include "jetstream/benchmark.hh"

namespace Jetstream {

Benchmark& Benchmark::getInstance() {
    static Benchmark instance;
    return instance;
}

U64 Benchmark::totalCount() {
    return benchmarks.size();
}

U64 Benchmark::currentCount() {
    return results.size();
}

void Benchmark::resetResults() {
    results.clear();
}

const Benchmark::ResultMapType& Benchmark::getResults() {
    return results;
}

void Benchmark::add(const std::string& module,
                    const std::string& device, 
                    const std::string& type, 
                    const BenchmarkFuncType& benchmark) {
    JST_TRACE("[BENCHMARK] Adding benchmark: {} - {} - {}", module, device, type);
    benchmarks[module].push_back({jst::fmt::format("{} - {} - ", device, type), benchmark});
}

void Benchmark::run(const std::string& outputType, std::ostream& out) {
    using namespace ankerl;
    using namespace std::chrono_literals;

    if (outputType != "markdown" && 
        outputType != "csv" && 
        outputType != "json" &&
        outputType != "quiet") {
        JST_FATAL("[BENCHMARK] Unknown output type: {}", outputType);
        JST_CHECK_THROW(Result::FATAL);
    }

    for (auto& [module, benchmark] : benchmarks) {
        using namespace std::chrono_literals;

        nanobench::Bench bench;

        bench.title(module)
             .output(nullptr)
             .timeUnit(1ms, "ms")
             .minEpochTime(100ms)
             .relative(false);

        if (outputType == "markdown") {
            bench.output(&out);
        }

        for (auto& [name, benchmark] : benchmark) {
            benchmark(bench, name);
        }

        if (outputType == "csv") {
            bench.render(nanobench::templates::csv(), out);
        } else if (outputType == "json") {
            bench.render(nanobench::templates::json(), out);
        }

        for (const auto& result : bench.results()) {
            const auto elapsed = result.median(nanobench::Result::Measure::elapsed);
            const auto error = result.medianAbsolutePercentError(nanobench::Result::Measure::elapsed);

            results[module].push_back({
                .name = result.config().mBenchmarkName,
                .ops_per_sec = bench.batch() / elapsed,
                .ms_per_op = elapsed / bench.batch() * 1000.0f,
                .error = error,
            });
        }
    }
}

}  // namespace Jetstream