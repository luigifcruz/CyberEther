#include "jetstream/benchmark.hh"
#include "jetstream/logger.hh"
#include "jetstream/registry.hh"
#include "jetstream/module.hh"
#include "jetstream/runtime.hh"
#include "jetstream/memory/tensor.hh"
#include "jetstream/memory/types.hh"

#include <chrono>
#include <cstring>

#include <nanobench.h>

namespace Jetstream {

struct Benchmark::Impl {
    void registerModuleBenchmarkSpecs(const std::string& moduleType,
                                      Benchmark::BenchmarkSpecFunc specFunc);
    void run(const std::string& outputType, std::ostream& out);
    U64 totalCount();
    U64 currentCount();
    void resetResults();
    const Benchmark::ResultMapType& getResults();
    const std::vector<Benchmark::ModuleBenchmarkSpecs>& listModuleBenchmarkSpecs();

    Benchmark::ResultMapType results;
    std::vector<Benchmark::ModuleBenchmarkSpecs> moduleBenchmarkSpecs;
};

Benchmark::Impl& Benchmark::benchmark() {
    static Impl impl;
    return impl;
}

void Benchmark::Impl::registerModuleBenchmarkSpecs(const std::string& moduleType,
                                                   Benchmark::BenchmarkSpecFunc specFunc) {
    JST_TRACE("[BENCHMARK] Registering benchmark specs for module type: {}", moduleType);
    moduleBenchmarkSpecs.push_back({
        .moduleType = moduleType,
        .specFunc = std::move(specFunc),
    });
}

void Benchmark::Impl::run(const std::string& outputType, std::ostream& out) {
    using namespace ankerl;
    using namespace std::chrono_literals;

    if (outputType != "markdown" &&
        outputType != "csv" &&
        outputType != "json" &&
        outputType != "quiet") {
        JST_FATAL("[BENCHMARK] Unknown output type: {}", outputType);
        JST_CHECK_THROW(Result::FATAL);
    }

    const int previousLogLevel = _JST_LOG_DEBUG_LEVEL();
    JST_LOG_SET_DEBUG_LEVEL(-1);

    resetResults();

    for (const auto& specEntry : moduleBenchmarkSpecs) {
        const auto specs = specEntry.specFunc();
        if (specs.empty()) {
            continue;
        }

        const auto implementations = Registry::ListAvailableModules(specEntry.moduleType);
        if (implementations.empty()) {
            JST_WARN("[BENCHMARK] No implementations found for module type: {}",
                     specEntry.moduleType);
            continue;
        }

        for (const auto& impl : implementations) {
            std::string moduleTitle = specEntry.moduleType;
            const auto blocks = Registry::ListAvailableBlocks(specEntry.moduleType);
            if (!blocks.empty()) {
                moduleTitle = blocks.front().title;
            }

            const std::string benchmarkTitle = jst::fmt::format("{} / {} / {} / {}",
                moduleTitle,
                GetDevicePrettyName(impl.device),
                GetRuntimePrettyName(impl.runtime),
                impl.provider);

            nanobench::Bench bench;

            bench.title(benchmarkTitle)
                 .output(nullptr)
                 .timeUnit(1ms, "ms")
                 .minEpochTime(100ms)
                 .relative(false);

            if (outputType == "markdown") {
                bench.output(&out);
            }

            for (const auto& spec : specs) {
                std::shared_ptr<Module> module;
                if (Registry::BuildModule(specEntry.moduleType, impl.device, impl.runtime, impl.provider, module) != Result::SUCCESS) {
                    continue;
                }

                TensorMap inputs;
                bool inputsValid = true;

                for (const auto& inputSpec : spec.inputs) {
                    Tensor tensor;
                    auto result = tensor.create(impl.device, inputSpec.dtype, inputSpec.shape);
                    if (result != Result::SUCCESS) {
                        inputsValid = false;
                        break;
                    }

                    std::memset(tensor.data(), 0, tensor.sizeBytes());

                    inputs[inputSpec.name] = {"benchmark", inputSpec.name, tensor};
                }

                if (!inputsValid) {
                    continue;
                }

                const std::string instanceName = jst::fmt::format("bench_{}", spec.variant);

                auto createResult = module->create(instanceName, spec.config, inputs);
                if (createResult != Result::SUCCESS) {
                    continue;
                }

                Runtime runtime("bench", impl.device, impl.runtime);
                if (runtime.create({{instanceName, module}}) != Result::SUCCESS) {
                    module->destroy();
                    continue;
                }

                const std::string benchName = spec.variant;

                bench.run(benchName, [&]() {
                    runtime.compute();
                });

                runtime.destroy();
                module->destroy();
            }

            if (outputType == "csv") {
                bench.render(nanobench::templates::csv(), out);
            } else if (outputType == "json") {
                bench.render(nanobench::templates::json(), out);
            }

            for (const auto& result : bench.results()) {
                const auto elapsed = result.median(nanobench::Result::Measure::elapsed);
                const auto error = result.medianAbsolutePercentError(
                    nanobench::Result::Measure::elapsed);

                results[benchmarkTitle].push_back({
                    .name = result.config().mBenchmarkName,
                    .opsPerSec = bench.batch() / elapsed,
                    .msPerOp = elapsed / bench.batch() * 1000.0,
                    .error = error,
                });
            }
        }
    }

    JST_LOG_SET_DEBUG_LEVEL(previousLogLevel);
}

U64 Benchmark::Impl::totalCount() {
    U64 count = 0;
    for (const auto& specEntry : moduleBenchmarkSpecs) {
        const auto implementations = Registry::ListAvailableModules(specEntry.moduleType);
        count += specEntry.specFunc().size() * implementations.size();
    }
    return count;
}

U64 Benchmark::Impl::currentCount() {
    U64 count = 0;
    for (const auto& [_, entries] : results) {
        count += entries.size();
    }
    return count;
}

void Benchmark::Impl::resetResults() {
    results.clear();
}

const Benchmark::ResultMapType& Benchmark::Impl::getResults() {
    return results;
}

const std::vector<Benchmark::ModuleBenchmarkSpecs>& Benchmark::Impl::listModuleBenchmarkSpecs() {
    return moduleBenchmarkSpecs;
}

void Benchmark::RegisterModuleBenchmarkSpecs(const std::string& moduleType,
                                             BenchmarkSpecFunc specFunc) {
    benchmark().registerModuleBenchmarkSpecs(moduleType, std::move(specFunc));
}

void Benchmark::Run(const std::string& outputType, std::ostream& out) {
    benchmark().run(outputType, out);
}

U64 Benchmark::TotalCount() {
    return benchmark().totalCount();
}

U64 Benchmark::CurrentCount() {
    return benchmark().currentCount();
}

void Benchmark::ResetResults() {
    benchmark().resetResults();
}

const Benchmark::ResultMapType& Benchmark::GetResults() {
    return benchmark().getResults();
}

const std::vector<Benchmark::ModuleBenchmarkSpecs>& Benchmark::ListModuleBenchmarkSpecs() {
    return benchmark().listModuleBenchmarkSpecs();
}

}  // namespace Jetstream
