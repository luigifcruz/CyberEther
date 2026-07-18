#include <catch2/catch_test_macros.hpp>

#include <any>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "jetstream/benchmark.hh"
#include "jetstream/detail/module_impl.hh"
#include "jetstream/logger.hh"
#include "jetstream/memory/types.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"

using namespace Jetstream;

namespace {

class RegistryCleanup {
 public:
    ~RegistryCleanup() {
        for (auto callback = callbacks.rbegin(); callback != callbacks.rend(); ++callback) {
            (*callback)();
        }
    }

    void module(const std::string& type,
                DeviceType device,
                RuntimeType runtime,
                const ProviderType& provider) {
        callbacks.emplace_back([=]() {
            (void)Registry::UnregisterModule(type, device, runtime, provider);
        });
    }

    void benchmark(const std::string& moduleType, const void* owner) {
        callbacks.emplace_back([=]() {
            (void)Registry::UnregisterBenchmark(moduleType, owner);
        });
    }

 private:
    std::vector<std::function<void()>> callbacks;
};

class DebugLevelGuard {
 public:
    DebugLevelGuard() : originalLevel(_JST_LOG_DEBUG_LEVEL()) {}

    ~DebugLevelGuard() {
        JST_LOG_SET_DEBUG_LEVEL(originalLevel);
    }

    int original() const {
        return originalLevel;
    }

 private:
    int originalLevel;
};

void PrepareRegistry() {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
}

std::string UniqueName(const std::string& label) {
    static std::atomic<std::uint64_t> sequence = 0;
    return "__ce_extensions_" + label + "_" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

Benchmark::Case MakeCase(const std::string& variant,
                         DataType dtype = DataType::F32,
                         U64 elements = 8) {
    Benchmark::Case benchmark;
    benchmark.variant = variant;
    benchmark.inputs.push_back({"signal", dtype, {elements}});
    benchmark.config["iterations"] = elements;
    return benchmark;
}

Registry::BenchmarkFactory MakeBenchmarkFactory(std::vector<Benchmark::Case> cases,
                                                 int* calls = nullptr) {
    return [cases = std::move(cases), calls]() {
        if (calls != nullptr) {
            ++*calls;
        }
        return cases;
    };
}

Registry::ModuleFactory MakeModuleFactory(int* calls = nullptr) {
    return [calls](const auto&, const auto&) -> std::shared_ptr<Module> {
        if (calls != nullptr) {
            ++*calls;
        }
        throw std::runtime_error("unexpected benchmark module construction");
    };
}

Benchmark::Case MakeRunnableCase(const std::string& variant) {
    Benchmark::Case benchmark;
    benchmark.variant = variant;
    return benchmark;
}

struct SyntheticModuleConfig final : Module::Config {
    std::string type() const override {
        return "__ce_extensions_benchmark_synthetic";
    }

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct SyntheticModuleState {
    std::vector<std::string> lifecycle;
    U64 computeCalls = 0;
    Result computeResult = Result::SUCCESS;
};

struct SyntheticModule final : Module::Impl,
                               NativeCpuRuntimeContext,
                               Scheduler::Context {
    Result create() override {
        state->lifecycle.emplace_back("module.create");
        return Result::SUCCESS;
    }

    Result destroy() override {
        state->lifecycle.emplace_back("module.destroy");
        return Result::SUCCESS;
    }

    Result computeInitialize() override {
        state->lifecycle.emplace_back("runtime.create");
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        if (state->computeCalls++ == 0) {
            state->lifecycle.emplace_back("runtime.compute");
        }
        return state->computeResult;
    }

    Result computeDeinitialize() override {
        state->lifecycle.emplace_back("runtime.destroy");
        return Result::SUCCESS;
    }

    std::shared_ptr<SyntheticModuleState> state;
};

Registry::ModuleFactory MakeRunnableModuleFactory(
    ProviderType provider,
    std::shared_ptr<SyntheticModuleState> state,
    int* calls = nullptr) {
    return [provider = std::move(provider), state = std::move(state), calls](
               const auto& environment,
               const auto& view) -> std::shared_ptr<Module> {
        if (calls != nullptr) {
            ++*calls;
        }

        const auto impl = std::make_shared<SyntheticModule>();
        impl->state = state;
        const auto runtimeContext = std::static_pointer_cast<Runtime::Context>(impl);
        const auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl);
        const auto context = std::make_shared<Module::Context>(runtimeContext,
                                                               schedulerContext,
                                                               environment,
                                                               view);
        return std::make_shared<Module>(DeviceType::CPU,
                                        RuntimeType::NATIVE,
                                        provider,
                                        impl,
                                        context,
                                        std::make_shared<SyntheticModuleConfig>(),
                                        std::make_shared<SyntheticModuleConfig>());
    };
}

void RequireCompletedLifecycle(const std::shared_ptr<SyntheticModuleState>& state) {
    const std::vector<std::string> expected{
        "module.create",
        "runtime.create",
        "runtime.compute",
        "runtime.destroy",
        "module.destroy",
    };
    REQUIRE(state->lifecycle == expected);
    REQUIRE(state->computeCalls > 0);
}

class ThrowingStreambuf final : public std::streambuf {
 protected:
    std::streamsize xsputn(const char*, std::streamsize) override {
        throw std::runtime_error("benchmark output failure");
    }

    int_type overflow(int_type) override {
        throw std::runtime_error("benchmark output failure");
    }
};

}  // namespace

TEST_CASE("Benchmark registry validates registrations",
          "[core][extensions][benchmark]") {
    PrepareRegistry();

    const auto type = UniqueName("validation");
    const auto factory = MakeBenchmarkFactory({});
    const int owner = 0;
    RegistryCleanup cleanup;
    cleanup.benchmark(type, &owner);

    REQUIRE(Registry::RegisterBenchmark("", factory, &owner) == Result::ERROR);
    REQUIRE(Registry::RegisterBenchmark(type, {}, &owner) == Result::ERROR);
    REQUIRE(Registry::RegisterBenchmark(type, factory, nullptr) == Result::ERROR);
    REQUIRE(Registry::UnregisterBenchmark(type, &owner) == Result::ERROR);
}

TEST_CASE("Benchmark registry preserves owner identity and registration order",
          "[core][extensions][benchmark]") {
    PrepareRegistry();

    const auto type = UniqueName("owners");
    const auto otherType = UniqueName("other");
    const int ownerA = 0;
    const int ownerB = 0;
    RegistryCleanup cleanup;
    cleanup.benchmark(type, &ownerA);
    cleanup.benchmark(type, &ownerB);
    cleanup.benchmark(otherType, &ownerA);

    REQUIRE(Registry::RegisterBenchmark(
                type,
                MakeBenchmarkFactory({MakeCase("small")}),
                &ownerA) == Result::SUCCESS);
    REQUIRE(Registry::RegisterBenchmark(
                type,
                MakeBenchmarkFactory({MakeCase("large", DataType::CF32, 1024)}),
                &ownerB) == Result::SUCCESS);
    REQUIRE(Registry::RegisterBenchmark(
                otherType,
                MakeBenchmarkFactory({MakeCase("other")}),
                &ownerA) == Result::SUCCESS);
    REQUIRE(Registry::RegisterBenchmark(
                type,
                MakeBenchmarkFactory({MakeCase("duplicate")}),
                &ownerA) == Result::ERROR);

    const auto registrations = Registry::ListAvailableBenchmarks(type);
    REQUIRE(registrations.size() == 2);
    REQUIRE(registrations[0].moduleType == type);
    REQUIRE(registrations[0].owner == &ownerA);
    REQUIRE(registrations[0].factory()[0].variant == "small");
    REQUIRE(registrations[1].owner == &ownerB);

    const auto large = registrations[1].factory();
    REQUIRE(large.size() == 1);
    REQUIRE(large[0].inputs[0].dtype == DataType::CF32);
    REQUIRE(std::any_cast<U64>(large[0].config.at("iterations")) == 1024);

    REQUIRE(Registry::UnregisterBenchmark(type, &ownerA) == Result::SUCCESS);
    const auto remaining = Registry::ListAvailableBenchmarks(type);
    REQUIRE(remaining.size() == 1);
    REQUIRE(remaining[0].owner == &ownerB);
    REQUIRE(Registry::UnregisterBenchmark(type, &ownerA) == Result::ERROR);
}

TEST_CASE("Benchmark total count multiplies test cases by registered implementations",
          "[core][extensions][benchmark]") {
    PrepareRegistry();

    const auto baseline = Benchmark::TotalCount();
    const auto type = UniqueName("count");
    const auto providerA = UniqueName("provider_a");
    const auto providerB = UniqueName("provider_b");
    const int owner = 0;
    int benchmarkFactoryCalls = 0;
    int moduleFactoryCalls = 0;
    RegistryCleanup cleanup;
    cleanup.benchmark(type, &owner);
    cleanup.module(type, DeviceType::CPU, RuntimeType::NATIVE, providerA);
    cleanup.module(type, DeviceType::CPU, RuntimeType::NATIVE, providerB);

    REQUIRE(Registry::RegisterBenchmark(
                type,
                MakeBenchmarkFactory({MakeCase("small"), MakeCase("large")},
                                     &benchmarkFactoryCalls),
                &owner) == Result::SUCCESS);
    REQUIRE(Registry::RegisterModule(type,
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     providerA,
                                     MakeModuleFactory(&moduleFactoryCalls)) == Result::SUCCESS);
    REQUIRE(Registry::RegisterModule(type,
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     providerB,
                                     MakeModuleFactory(&moduleFactoryCalls)) == Result::SUCCESS);

    REQUIRE(Benchmark::TotalCount() == baseline + 4);
    REQUIRE(benchmarkFactoryCalls == 1);
    REQUIRE(moduleFactoryCalls == 0);

    REQUIRE(Registry::UnregisterModule(
                type, DeviceType::CPU, RuntimeType::NATIVE, providerB) == Result::SUCCESS);
    REQUIRE(Benchmark::TotalCount() == baseline + 2);
    REQUIRE(benchmarkFactoryCalls == 2);
    REQUIRE(moduleFactoryCalls == 0);

    REQUIRE(Registry::UnregisterBenchmark(type, &owner) == Result::SUCCESS);
    REQUIRE(Registry::UnregisterModule(
                type, DeviceType::CPU, RuntimeType::NATIVE, providerA) == Result::SUCCESS);
    REQUIRE(Benchmark::TotalCount() == baseline);
}

TEST_CASE("Benchmark runs synthetic modules and renders supported formats",
          "[core][extensions][benchmark]") {
    PrepareRegistry();

    for (const std::string format : {"markdown", "csv", "json"}) {
        CAPTURE(format);

        const auto type = UniqueName("run_" + format);
        const auto provider = UniqueName("provider");
        const std::string variant = "tiny";
        const int owner = 0;
        int moduleFactoryCalls = 0;
        auto state = std::make_shared<SyntheticModuleState>();
        RegistryCleanup cleanup;
        DebugLevelGuard debugLevel;
        cleanup.benchmark(type, &owner);
        cleanup.module(type, DeviceType::CPU, RuntimeType::NATIVE, provider);

        REQUIRE(Registry::RegisterBenchmark(
                    type,
                    MakeBenchmarkFactory({MakeRunnableCase(variant)}),
                    &owner) == Result::SUCCESS);
        REQUIRE(Registry::RegisterModule(
                    type,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    provider,
                    MakeRunnableModuleFactory(provider, state, &moduleFactoryCalls)) ==
                Result::SUCCESS);

        std::ostringstream output;
        Benchmark::Run(format, type, output);

        const auto rendered = output.str();
        REQUIRE(!rendered.empty());
        REQUIRE(rendered.find(variant) != std::string::npos);
        if (format == "markdown") {
            REQUIRE(rendered.find('|') != std::string::npos);
        } else if (format == "csv") {
            REQUIRE(rendered.find("\"title\";\"name\"") != std::string::npos);
        } else {
            REQUIRE(rendered.find("\"results\": [") != std::string::npos);
        }

        REQUIRE(moduleFactoryCalls == 1);
        RequireCompletedLifecycle(state);
        REQUIRE(_JST_LOG_DEBUG_LEVEL() == debugLevel.original());

        const auto& results = Benchmark::GetResults();
        REQUIRE(!results.empty());
        REQUIRE(Benchmark::CurrentCount() == 1);
        const auto title = type + " / CPU / Native / " + provider;
        const auto result = results.find(title);
        REQUIRE(result != results.end());
        REQUIRE(result->second.size() == 1);
        REQUIRE(result->second[0].name == variant);
        REQUIRE(result->second[0].opsPerSec > 0.0);
        REQUIRE(result->second[0].msPerOp >= 0.0);

        Benchmark::ResetResults();
        REQUIRE(Benchmark::CurrentCount() == 0);
        REQUIRE(Benchmark::GetResults().empty());
    }
}

TEST_CASE("Benchmark orchestration filters factories without running implementations",
          "[core][extensions][benchmark]") {
    PrepareRegistry();
    Benchmark::ResetResults();

    const auto selectedType = UniqueName("selected");
    const auto otherType = UniqueName("unselected");
    const int selectedOwner = 0;
    const int otherOwner = 0;
    int selectedCalls = 0;
    int otherCalls = 0;
    RegistryCleanup cleanup;
    cleanup.benchmark(selectedType, &selectedOwner);
    cleanup.benchmark(otherType, &otherOwner);

    REQUIRE(Registry::RegisterBenchmark(
                selectedType,
                MakeBenchmarkFactory({MakeCase("small"), MakeCase("large")}, &selectedCalls),
                &selectedOwner) == Result::SUCCESS);
    REQUIRE(Registry::RegisterBenchmark(
                otherType,
                MakeBenchmarkFactory({MakeCase("other")}, &otherCalls),
                &otherOwner) == Result::SUCCESS);

    std::ostringstream output;
    Benchmark::Run("quiet", selectedType, output);

    REQUIRE(selectedCalls == 1);
    REQUIRE(otherCalls == 0);
    REQUIRE(output.str().empty());
    REQUIRE(Benchmark::CurrentCount() == 0);
    REQUIRE(Benchmark::GetResults().empty());
}

TEST_CASE("Benchmark orchestration skips empty factories before module construction",
          "[core][extensions][benchmark]") {
    PrepareRegistry();
    Benchmark::ResetResults();

    const auto type = UniqueName("empty");
    const auto provider = UniqueName("provider");
    const int owner = 0;
    int benchmarkFactoryCalls = 0;
    int moduleFactoryCalls = 0;
    RegistryCleanup cleanup;
    cleanup.benchmark(type, &owner);
    cleanup.module(type, DeviceType::CPU, RuntimeType::NATIVE, provider);

    REQUIRE(Registry::RegisterBenchmark(
                type,
                MakeBenchmarkFactory({}, &benchmarkFactoryCalls),
                &owner) == Result::SUCCESS);
    REQUIRE(Registry::RegisterModule(type,
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     provider,
                                     MakeModuleFactory(&moduleFactoryCalls)) == Result::SUCCESS);

    std::ostringstream output;
    Benchmark::Run("quiet", type, output);

    REQUIRE(benchmarkFactoryCalls == 1);
    REQUIRE(moduleFactoryCalls == 0);
    REQUIRE(output.str().empty());
    REQUIRE(Benchmark::GetResults().empty());
}

TEST_CASE("Benchmark handles registry module factory failures",
          "[core][extensions][benchmark]") {
    PrepareRegistry();

    SECTION("null module returned by a factory is rejected") {
        const auto type = UniqueName("null_module");
        const auto provider = UniqueName("provider");
        RegistryCleanup cleanup;
        cleanup.module(type, DeviceType::CPU, RuntimeType::NATIVE, provider);

        REQUIRE(Registry::RegisterModule(
                    type,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    provider,
                    [](const auto&, const auto&) -> std::shared_ptr<Module> {
                        return nullptr;
                    }) == Result::SUCCESS);

        std::shared_ptr<Module> module;
        CHECK(Registry::BuildModule(type,
                                    DeviceType::CPU,
                                    RuntimeType::NATIVE,
                                    provider,
                                    module) == Result::ERROR);
        REQUIRE(module == nullptr);
    }

    SECTION("throwing module factory propagates and restores the logger level") {
        Benchmark::ResetResults();
        const auto type = UniqueName("throwing_module");
        const auto provider = UniqueName("provider");
        const int owner = 0;
        RegistryCleanup cleanup;
        DebugLevelGuard debugLevel;
        cleanup.benchmark(type, &owner);
        cleanup.module(type, DeviceType::CPU, RuntimeType::NATIVE, provider);

        REQUIRE(Registry::RegisterBenchmark(
                    type,
                    MakeBenchmarkFactory({MakeRunnableCase("unused")}),
                    &owner) == Result::SUCCESS);
        REQUIRE(Registry::RegisterModule(
                    type,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    provider,
                    [](const auto&, const auto&) -> std::shared_ptr<Module> {
                        throw std::runtime_error("module factory failure");
                    }) == Result::SUCCESS);

        std::ostringstream output;
        REQUIRE_THROWS_AS(Benchmark::Run("quiet", type, output), std::runtime_error);
        REQUIRE(_JST_LOG_DEBUG_LEVEL() == debugLevel.original());
        REQUIRE(Benchmark::CurrentCount() == 0);
        REQUIRE(Benchmark::GetResults().empty());
    }
}

TEST_CASE("Benchmark does not publish failed runtime computations",
          "[core][extensions][benchmark]") {
    PrepareRegistry();
    Benchmark::ResetResults();

    const auto type = UniqueName("compute_failure");
    const auto provider = UniqueName("provider");
    const int owner = 0;
    auto state = std::make_shared<SyntheticModuleState>();
    state->computeResult = Result::ERROR;
    RegistryCleanup cleanup;
    cleanup.benchmark(type, &owner);
    cleanup.module(type, DeviceType::CPU, RuntimeType::NATIVE, provider);

    REQUIRE(Registry::RegisterBenchmark(
                type,
                MakeBenchmarkFactory({MakeRunnableCase("failure")}),
                &owner) == Result::SUCCESS);
    REQUIRE(Registry::RegisterModule(
                type,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                provider,
                MakeRunnableModuleFactory(provider, state)) == Result::SUCCESS);

    std::ostringstream output;
    Benchmark::Run("quiet", type, output);

    RequireCompletedLifecycle(state);
    REQUIRE(output.str().empty());
    // Current defect: Benchmark::Run records measurements after compute reports failure.
    CHECK(Benchmark::CurrentCount() == 0);
    CHECK(Benchmark::GetResults().empty());
    Benchmark::ResetResults();
}

TEST_CASE("Benchmark orchestration validates output and restores process state",
          "[core][extensions][benchmark]") {
    PrepareRegistry();

    SECTION("unknown output type is rejected before invoking factories") {
        const auto type = UniqueName("output");
        const int owner = 0;
        int factoryCalls = 0;
        RegistryCleanup cleanup;
        cleanup.benchmark(type, &owner);
        REQUIRE(Registry::RegisterBenchmark(
                    type,
                    MakeBenchmarkFactory({MakeCase("unused")}, &factoryCalls),
                    &owner) == Result::SUCCESS);

        std::ostringstream output;
        REQUIRE_THROWS(Benchmark::Run("xml", type, output));
        REQUIRE(factoryCalls == 0);
        REQUIRE(output.str().empty());
    }

    SECTION("throwing factory restores the logger level") {
        const auto type = UniqueName("throwing");
        const int owner = 0;
        RegistryCleanup cleanup;
        DebugLevelGuard debugLevel;
        cleanup.benchmark(type, &owner);
        REQUIRE(Registry::RegisterBenchmark(
                    type,
                    []() -> std::vector<Benchmark::Case> {
                        throw std::runtime_error("benchmark fixture failure");
                    },
                    &owner) == Result::SUCCESS);

        std::ostringstream output;
        REQUIRE_THROWS_AS(Benchmark::Run("quiet", type, output), std::runtime_error);
        REQUIRE(_JST_LOG_DEBUG_LEVEL() == debugLevel.original());
        REQUIRE(Benchmark::GetResults().empty());
    }

    SECTION("output exception after teardown restores the logger level") {
        Benchmark::ResetResults();
        const auto type = UniqueName("throwing_output");
        const auto provider = UniqueName("provider");
        const int owner = 0;
        auto state = std::make_shared<SyntheticModuleState>();
        RegistryCleanup cleanup;
        DebugLevelGuard debugLevel;
        cleanup.benchmark(type, &owner);
        cleanup.module(type, DeviceType::CPU, RuntimeType::NATIVE, provider);

        REQUIRE(Registry::RegisterBenchmark(
                    type,
                    MakeBenchmarkFactory({MakeRunnableCase("output")}),
                    &owner) == Result::SUCCESS);
        REQUIRE(Registry::RegisterModule(
                    type,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    provider,
                    MakeRunnableModuleFactory(provider, state)) == Result::SUCCESS);

        ThrowingStreambuf buffer;
        std::ostream output(&buffer);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        // CSV rendering happens only after runtime and module teardown.
        REQUIRE_THROWS(Benchmark::Run("csv", type, output));

        RequireCompletedLifecycle(state);
        REQUIRE(_JST_LOG_DEBUG_LEVEL() == debugLevel.original());
        REQUIRE(Benchmark::CurrentCount() == 0);
        REQUIRE(Benchmark::GetResults().empty());
    }
}
