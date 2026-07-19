#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <barrier>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "jetstream/detail/block_impl.hh"
#include "jetstream/detail/module_impl.hh"
#include "jetstream/flowgraph_environment.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

namespace {

class RegistryCleanup {
 public:
    ~RegistryCleanup() {
        (void)Registry::DiscardStaticRegistrations();
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

    void block(const std::string& type) {
        callbacks.emplace_back([=]() {
            (void)Registry::UnregisterBlock(type);
        });
    }

    void flowgraph(const std::string& key) {
        callbacks.emplace_back([=]() {
            (void)Registry::UnregisterFlowgraph(key);
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

void PrepareRegistry() {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
}

std::shared_ptr<Module> MakeTestModule() {
    return std::make_shared<Module>(DeviceType::CPU,
                                    RuntimeType::NATIVE,
                                    "__ce_registry_test_fixture_provider",
                                    std::make_shared<Module::Impl>(),
                                    nullptr,
                                    nullptr,
                                    nullptr);
}

std::shared_ptr<Block> MakeTestBlock() {
    return std::make_shared<Block>(std::make_shared<Block::Impl>(), nullptr, nullptr);
}

Registry::FlowgraphRegistration MakeFlowgraph(const std::string& key,
                                               const std::string& title) {
    Registry::FlowgraphRegistration record;
    record.key = key;
    record.title = title;
    record.summary = "Registry test summary";
    record.description = "Registry test description";
    record.content = "---\nversion: 2\ngraph: []\n";
    return record;
}

Registry::BenchmarkFactory MakeBenchmarkFactory(const std::string& variant) {
    return [variant]() {
        Benchmark::Case benchmark;
        benchmark.variant = variant;
        return std::vector<Benchmark::Case>{benchmark};
    };
}

JST_BENCHMARKS("__ce_registry_test_tu_static_benchmark") {
    Benchmark::Case benchmark;
    benchmark.variant = "tu-static";
    return {benchmark};
}

}  // namespace

TEST_CASE("Registry validates direct registrations", "[core][registry][validation]") {
    PrepareRegistry();

    const Registry::ModuleFactory moduleFactory = [](const auto&, const auto&) {
        return MakeTestModule();
    };
    const Registry::BlockFactory blockFactory = []() {
        return MakeTestBlock();
    };
    int benchmarkOwner = 0;
    RegistryCleanup cleanup;

    cleanup.module("", DeviceType::CPU, RuntimeType::NATIVE, "__ce_registry_test_provider");
    REQUIRE(Registry::RegisterModule("",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     "__ce_registry_test_provider",
                                     moduleFactory) == Result::ERROR);

    const std::string noDevice = "__ce_registry_test_validation_no_device";
    cleanup.module(noDevice, DeviceType::None, RuntimeType::NATIVE, "__ce_registry_test_provider");
    REQUIRE(Registry::RegisterModule(noDevice,
                                     DeviceType::None,
                                     RuntimeType::NATIVE,
                                     "__ce_registry_test_provider",
                                     moduleFactory) == Result::ERROR);

    const std::string noRuntime = "__ce_registry_test_validation_no_runtime";
    cleanup.module(noRuntime, DeviceType::CPU, RuntimeType::NONE, "__ce_registry_test_provider");
    REQUIRE(Registry::RegisterModule(noRuntime,
                                     DeviceType::CPU,
                                     RuntimeType::NONE,
                                     "__ce_registry_test_provider",
                                     moduleFactory) == Result::ERROR);

    const std::string noProvider = "__ce_registry_test_validation_no_provider";
    cleanup.module(noProvider, DeviceType::CPU, RuntimeType::NATIVE, "");
    REQUIRE(Registry::RegisterModule(noProvider,
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     "",
                                     moduleFactory) == Result::ERROR);

    const std::string noModuleFactory = "__ce_registry_test_validation_no_module_factory";
    cleanup.module(noModuleFactory,
                   DeviceType::CPU,
                   RuntimeType::NATIVE,
                   "__ce_registry_test_provider");
    CHECK(Registry::RegisterModule(noModuleFactory,
                                   DeviceType::CPU,
                                   RuntimeType::NATIVE,
                                   "__ce_registry_test_provider",
                                   {}) == Result::ERROR);
    REQUIRE(Registry::ListAvailableModules(noModuleFactory).empty());
    REQUIRE(Registry::RegisterModule(noModuleFactory,
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     "__ce_registry_test_provider",
                                     moduleFactory) == Result::SUCCESS);

    cleanup.block("");
    REQUIRE(Registry::RegisterBlock("",
                                    "__ce_registry_test_domain",
                                    "Title",
                                    "Summary",
                                    "Description",
                                    blockFactory) == Result::ERROR);

    const std::string noDomain = "__ce_registry_test_validation_no_domain";
    cleanup.block(noDomain);
    REQUIRE(Registry::RegisterBlock(noDomain,
                                    "",
                                    "Title",
                                    "Summary",
                                    "Description",
                                    blockFactory) == Result::ERROR);

    const std::string noBlockFactory = "__ce_registry_test_validation_no_block_factory";
    cleanup.block(noBlockFactory);
    CHECK(Registry::RegisterBlock(noBlockFactory,
                                  "__ce_registry_test_domain",
                                  "Title",
                                  "Summary",
                                  "Description",
                                  {}) == Result::ERROR);
    REQUIRE(Registry::ListAvailableBlocks(noBlockFactory).empty());
    REQUIRE(Registry::RegisterBlock(noBlockFactory,
                                    "__ce_registry_test_domain",
                                    "Title",
                                    "Summary",
                                    "Description",
                                    blockFactory) == Result::SUCCESS);
    CHECK(Registry::RegisterBlock(noBlockFactory,
                                  "__ce_registry_test_domain",
                                  "Title",
                                  "Summary",
                                  "Description",
                                  {}) == Result::ERROR);

    auto emptyFlowgraph = MakeFlowgraph("", "Empty key");
    cleanup.flowgraph("");
    REQUIRE(Registry::RegisterFlowgraph("", emptyFlowgraph) == Result::ERROR);

    const std::string argumentKey = "__ce_registry_test_validation_argument_key";
    const std::string metadataKey = "__ce_registry_test_validation_metadata_key";
    auto mismatchedFlowgraph = MakeFlowgraph(metadataKey, "Mismatched key");
    cleanup.flowgraph(argumentKey);
    cleanup.flowgraph(metadataKey);
    CHECK(Registry::RegisterFlowgraph(argumentKey, mismatchedFlowgraph) == Result::ERROR);

    cleanup.benchmark("", &benchmarkOwner);
    REQUIRE(Registry::RegisterBenchmark("",
                                        MakeBenchmarkFactory("empty-module"),
                                        &benchmarkOwner) == Result::ERROR);

    const std::string noBenchmarkFactory = "__ce_registry_test_validation_no_benchmark_factory";
    cleanup.benchmark(noBenchmarkFactory, &benchmarkOwner);
    REQUIRE(Registry::RegisterBenchmark(noBenchmarkFactory,
                                        {},
                                        &benchmarkOwner) == Result::ERROR);

    const std::string noBenchmarkOwner = "__ce_registry_test_validation_no_benchmark_owner";
    cleanup.benchmark(noBenchmarkOwner, nullptr);
    REQUIRE(Registry::RegisterBenchmark(noBenchmarkOwner,
                                        MakeBenchmarkFactory("no-owner"),
                                        nullptr) == Result::ERROR);
}

TEST_CASE("Registry registers filters builds and unregisters modules", "[core][registry][module]") {
    PrepareRegistry();

    const std::string type = "__ce_registry_test_direct_module";
    const std::string otherType = "__ce_registry_test_direct_module_other";
    const ProviderType nativeProvider = "__ce_registry_test_native_provider";
    const ProviderType pythonProvider = "__ce_registry_test_python_provider";
    auto nativeModule = MakeTestModule();
    auto pythonModule = MakeTestModule();
    auto otherModule = MakeTestModule();
    auto environment = std::make_shared<Flowgraph::Environment>(
        std::shared_ptr<Flowgraph::Impl>{});
    auto view = std::make_shared<Flowgraph::View>(std::shared_ptr<Flowgraph::Impl>{});
    std::shared_ptr<Flowgraph::Environment> seenEnvironment;
    std::shared_ptr<Flowgraph::View> seenView;
    int nativeFactoryCalls = 0;
    RegistryCleanup cleanup;
    cleanup.module(type, DeviceType::CPU, RuntimeType::NATIVE, nativeProvider);
    cleanup.module(type, DeviceType::CUDA, RuntimeType::PYTHON, pythonProvider);
    cleanup.module(otherType, DeviceType::CPU, RuntimeType::NATIVE, nativeProvider);

    REQUIRE(Registry::RegisterModule(
                type,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                nativeProvider,
                [&](const auto& factoryEnvironment, const auto& factoryView) {
                    ++nativeFactoryCalls;
                    seenEnvironment = factoryEnvironment;
                    seenView = factoryView;
                    return nativeModule;
                }) == Result::SUCCESS);
    REQUIRE(Registry::RegisterModule(
                type,
                DeviceType::CUDA,
                RuntimeType::PYTHON,
                pythonProvider,
                [&](const auto&, const auto&) {
                    return pythonModule;
                }) == Result::SUCCESS);
    REQUIRE(Registry::RegisterModule(
                otherType,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                nativeProvider,
                [&](const auto&, const auto&) {
                    return otherModule;
                }) == Result::SUCCESS);

    REQUIRE(Registry::RegisterModule(
                type,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                nativeProvider,
                [](const auto&, const auto&) {
                    return MakeTestModule();
                }) == Result::ERROR);

    auto byType = Registry::ListAvailableModules(type);
    REQUIRE(byType.size() == 2);
    REQUIRE(byType[0].device == DeviceType::CPU);
    REQUIRE(byType[0].runtime == RuntimeType::NATIVE);
    REQUIRE(byType[0].provider == nativeProvider);
    REQUIRE(static_cast<bool>(byType[0].factory));
    REQUIRE(byType[1].device == DeviceType::CUDA);
    REQUIRE(byType[1].runtime == RuntimeType::PYTHON);
    REQUIRE(byType[1].provider == pythonProvider);

    const auto byDevice = Registry::ListAvailableModules(type, DeviceType::CPU);
    REQUIRE(byDevice.size() == 1);
    REQUIRE(byDevice[0].provider == nativeProvider);

    const auto byRuntime = Registry::ListAvailableModules(type, std::nullopt, RuntimeType::PYTHON);
    REQUIRE(byRuntime.size() == 1);
    REQUIRE(byRuntime[0].provider == pythonProvider);

    const auto byProvider = Registry::ListAvailableModules(type,
                                                            std::nullopt,
                                                            std::nullopt,
                                                            pythonProvider);
    REQUIRE(byProvider.size() == 1);
    REQUIRE(byProvider[0].device == DeviceType::CUDA);

    const auto byAllFields = Registry::ListAvailableModules(type,
                                                             DeviceType::CPU,
                                                             RuntimeType::NATIVE,
                                                             nativeProvider);
    REQUIRE(byAllFields.size() == 1);
    REQUIRE(byAllFields[0].type == type);
    REQUIRE(Registry::ListAvailableModules(type,
                                           DeviceType::CPU,
                                           RuntimeType::PYTHON,
                                           nativeProvider).empty());

    const auto acrossTypes = Registry::ListAvailableModules("",
                                                             DeviceType::CPU,
                                                             RuntimeType::NATIVE,
                                                             nativeProvider);
    REQUIRE(acrossTypes.size() == 2);

    byType[0].provider = "__ce_registry_test_snapshot_mutation";
    REQUIRE(Registry::ListAvailableModules(type, DeviceType::CPU)[0].provider == nativeProvider);
    REQUIRE(Registry::ListAvailableModules("__ce_registry_test_missing_module").empty());

    std::shared_ptr<Module> built;
    REQUIRE(Registry::BuildModule(type,
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  nativeProvider,
                                  built,
                                  environment,
                                  view) == Result::SUCCESS);
    REQUIRE(built == nativeModule);
    REQUIRE(nativeFactoryCalls == 1);
    REQUIRE(seenEnvironment == environment);
    REQUIRE(seenView == view);

    auto untouched = otherModule;
    REQUIRE(Registry::BuildModule(type,
                                  DeviceType::None,
                                  RuntimeType::NATIVE,
                                  nativeProvider,
                                  untouched) == Result::ERROR);
    REQUIRE(untouched == otherModule);
    REQUIRE(Registry::BuildModule(type,
                                  DeviceType::CPU,
                                  RuntimeType::NONE,
                                  nativeProvider,
                                  untouched) == Result::ERROR);
    REQUIRE(untouched == otherModule);
    REQUIRE(Registry::BuildModule(type,
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  "",
                                  untouched) == Result::ERROR);
    REQUIRE(untouched == otherModule);
    REQUIRE(Registry::BuildModule(type,
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  "__ce_registry_test_missing_provider",
                                  untouched) == Result::ERROR);
    REQUIRE(untouched == otherModule);

    REQUIRE(Registry::UnregisterModule(type,
                                       DeviceType::CPU,
                                       RuntimeType::NATIVE,
                                       nativeProvider) == Result::SUCCESS);
    REQUIRE(Registry::ListAvailableModules(type).size() == 1);
    REQUIRE(Registry::UnregisterModule(type,
                                       DeviceType::CPU,
                                       RuntimeType::NATIVE,
                                       nativeProvider) == Result::ERROR);
}

TEST_CASE("Registry registers builds and unregisters blocks", "[core][registry][block]") {
    PrepareRegistry();

    const std::string type = "__ce_registry_test_direct_block";
    auto expectedBlock = MakeTestBlock();
    int originalFactoryCalls = 0;
    int replacementFactoryCalls = 0;
    RegistryCleanup cleanup;
    cleanup.block(type);

    REQUIRE(Registry::RegisterBlock(type,
                                    "__ce_registry_test_domain",
                                    "Registry Fixture",
                                    "Fixture summary",
                                    "Fixture description",
                                    [&]() {
                                        ++originalFactoryCalls;
                                        return expectedBlock;
                                    }) == Result::SUCCESS);

    REQUIRE(Registry::RegisterBlock(type,
                                    "__ce_registry_test_domain",
                                    "Registry Fixture",
                                    "Fixture summary",
                                    "Fixture description",
                                    [&]() {
                                        ++replacementFactoryCalls;
                                        return MakeTestBlock();
                                    }) == Result::SUCCESS);
    REQUIRE(Registry::RegisterBlock(type,
                                    "__ce_registry_test_domain",
                                    "Conflicting title",
                                    "Fixture summary",
                                    "Fixture description",
                                    []() {
                                        return MakeTestBlock();
                                    }) == Result::ERROR);
    REQUIRE(Registry::RegisterBlock(type,
                                    "__ce_registry_test_other_domain",
                                    "Registry Fixture",
                                    "Fixture summary",
                                    "Fixture description",
                                    []() {
                                        return MakeTestBlock();
                                    }) == Result::ERROR);
    REQUIRE(Registry::RegisterBlock(type,
                                    "__ce_registry_test_domain",
                                    "Registry Fixture",
                                    "Conflicting summary",
                                    "Fixture description",
                                    []() {
                                        return MakeTestBlock();
                                    }) == Result::ERROR);
    REQUIRE(Registry::RegisterBlock(type,
                                    "__ce_registry_test_domain",
                                    "Registry Fixture",
                                    "Fixture summary",
                                    "Conflicting description",
                                    []() {
                                        return MakeTestBlock();
                                    }) == Result::ERROR);

    auto registrations = Registry::ListAvailableBlocks(type);
    REQUIRE(registrations.size() == 1);
    REQUIRE(registrations[0].type == type);
    REQUIRE(registrations[0].domain == "__ce_registry_test_domain");
    REQUIRE(registrations[0].title == "Registry Fixture");
    REQUIRE(registrations[0].summary == "Fixture summary");
    REQUIRE(registrations[0].description == "Fixture description");
    REQUIRE(static_cast<bool>(registrations[0].factory));

    registrations[0].title = "Snapshot mutation";
    REQUIRE(Registry::ListAvailableBlocks(type)[0].title == "Registry Fixture");
    REQUIRE(Registry::ListAvailableBlocks("__ce_registry_test_missing_block").empty());

    std::shared_ptr<Block> built;
    REQUIRE(Registry::BuildBlock(type, built) == Result::SUCCESS);
    REQUIRE(built == expectedBlock);
    REQUIRE(originalFactoryCalls == 1);
    REQUIRE(replacementFactoryCalls == 0);

    REQUIRE(Registry::UnregisterBlock(type) == Result::SUCCESS);
    REQUIRE(Registry::ListAvailableBlocks(type).empty());

    auto untouched = expectedBlock;
    REQUIRE(Registry::BuildBlock(type, untouched) == Result::ERROR);
    REQUIRE(untouched == expectedBlock);
    REQUIRE(Registry::UnregisterBlock(type) == Result::ERROR);
}

TEST_CASE("Registry contains invalid module and block factory results",
          "[core][registry][factory][errors]") {
    PrepareRegistry();

    const std::string throwingModuleType = "__ce_registry_test_throwing_module_factory";
    const std::string nullModuleType = "__ce_registry_test_null_module_factory";
    const ProviderType provider = "__ce_registry_test_invalid_factory_provider";
    const std::string throwingBlockType = "__ce_registry_test_throwing_block_factory";
    const std::string nullBlockType = "__ce_registry_test_null_block_factory";
    RegistryCleanup cleanup;
    cleanup.module(throwingModuleType, DeviceType::CPU, RuntimeType::NATIVE, provider);
    cleanup.module(nullModuleType, DeviceType::CPU, RuntimeType::NATIVE, provider);
    cleanup.block(throwingBlockType);
    cleanup.block(nullBlockType);

    REQUIRE(Registry::RegisterModule(
                throwingModuleType,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                provider,
                [](const auto&, const auto&) -> std::shared_ptr<Module> {
                    throw std::runtime_error("throwing module factory");
                }) == Result::SUCCESS);
    REQUIRE(Registry::RegisterModule(
                nullModuleType,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                provider,
                [](const auto&, const auto&) -> std::shared_ptr<Module> {
                    return nullptr;
                }) == Result::SUCCESS);
    REQUIRE(Registry::RegisterBlock(throwingBlockType,
                                    "__ce_registry_test_domain",
                                    "Throwing Block Factory",
                                    "Throwing factory summary",
                                    "Throwing factory description",
                                    []() -> std::shared_ptr<Block> {
                                        throw std::runtime_error("throwing block factory");
                                    }) == Result::SUCCESS);
    REQUIRE(Registry::RegisterBlock(nullBlockType,
                                    "__ce_registry_test_domain",
                                    "Null Block Factory",
                                    "Null factory summary",
                                    "Null factory description",
                                    []() -> std::shared_ptr<Block> {
                                        return nullptr;
                                    }) == Result::SUCCESS);

    auto moduleSentinel = MakeTestModule();
    auto throwingModule = moduleSentinel;
    Result throwingModuleResult = Result::ERROR;
    bool moduleExceptionEscaped = false;
    try {
        throwingModuleResult = Registry::BuildModule(throwingModuleType,
                                                      DeviceType::CPU,
                                                      RuntimeType::NATIVE,
                                                      provider,
                                                      throwingModule);
    } catch (...) {
        moduleExceptionEscaped = true;
    }
    // Expected to fail currently: module factory exceptions escape BuildModule.
    CHECK_FALSE(moduleExceptionEscaped);
    CHECK(throwingModuleResult == Result::ERROR);
    CHECK(throwingModule == moduleSentinel);

    auto nullModule = moduleSentinel;
    const auto nullModuleResult = Registry::BuildModule(nullModuleType,
                                                        DeviceType::CPU,
                                                        RuntimeType::NATIVE,
                                                        provider,
                                                        nullModule);
    CHECK(nullModuleResult == Result::ERROR);
    CHECK(nullModule == moduleSentinel);

    auto blockSentinel = MakeTestBlock();
    auto throwingBlock = blockSentinel;
    Result throwingBlockResult = Result::ERROR;
    bool blockExceptionEscaped = false;
    try {
        throwingBlockResult = Registry::BuildBlock(throwingBlockType, throwingBlock);
    } catch (...) {
        blockExceptionEscaped = true;
    }
    // Expected to fail currently: block factory exceptions escape BuildBlock.
    CHECK_FALSE(blockExceptionEscaped);
    CHECK(throwingBlockResult == Result::ERROR);
    CHECK(throwingBlock == blockSentinel);

    auto nullBlock = blockSentinel;
    const auto nullBlockResult = Registry::BuildBlock(nullBlockType, nullBlock);
    CHECK(nullBlockResult == Result::ERROR);
    CHECK(nullBlock == blockSentinel);
}

TEST_CASE("Registry lists and unregisters flowgraphs and benchmarks", "[core][registry][metadata]") {
    PrepareRegistry();

    const std::string flowgraphKey = "__ce_registry_test_direct_flowgraph";
    const std::string benchmarkType = "__ce_registry_test_direct_benchmark";
    const std::string otherBenchmarkType = "__ce_registry_test_direct_benchmark_other";
    int ownerA = 0;
    int ownerB = 0;
    RegistryCleanup cleanup;
    cleanup.flowgraph(flowgraphKey);
    cleanup.benchmark(benchmarkType, &ownerA);
    cleanup.benchmark(benchmarkType, &ownerB);
    cleanup.benchmark(otherBenchmarkType, &ownerA);

    auto flowgraph = MakeFlowgraph(flowgraphKey, "Registry Flowgraph");
    REQUIRE(Registry::RegisterFlowgraph(flowgraphKey, flowgraph) == Result::SUCCESS);
    REQUIRE(Registry::RegisterFlowgraph(flowgraphKey, flowgraph) == Result::ERROR);
    flowgraph.title = "Caller mutation";
    flowgraph.content = "caller mutation";

    const auto flowgraphs = Registry::ListAvailableFlowgraphs(flowgraphKey);
    REQUIRE(flowgraphs.size() == 1);
    REQUIRE(flowgraphs[0].key == flowgraphKey);
    REQUIRE(flowgraphs[0].title == "Registry Flowgraph");
    REQUIRE(flowgraphs[0].summary == "Registry test summary");
    REQUIRE(flowgraphs[0].description == "Registry test description");
    REQUIRE(flowgraphs[0].content == "---\nversion: 2\ngraph: []\n");
    REQUIRE(Registry::ListAvailableFlowgraphs("__ce_registry_test_missing_flowgraph").empty());

    REQUIRE(Registry::RegisterBenchmark(benchmarkType,
                                        MakeBenchmarkFactory("owner-a"),
                                        &ownerA) == Result::SUCCESS);
    REQUIRE(Registry::RegisterBenchmark(benchmarkType,
                                        MakeBenchmarkFactory("owner-b"),
                                        &ownerB) == Result::SUCCESS);
    REQUIRE(Registry::RegisterBenchmark(benchmarkType,
                                        MakeBenchmarkFactory("duplicate"),
                                        &ownerA) == Result::ERROR);
    REQUIRE(Registry::RegisterBenchmark(otherBenchmarkType,
                                        MakeBenchmarkFactory("other-type"),
                                        &ownerA) == Result::SUCCESS);

    const auto benchmarks = Registry::ListAvailableBenchmarks(benchmarkType);
    REQUIRE(benchmarks.size() == 2);
    REQUIRE(benchmarks[0].moduleType == benchmarkType);
    REQUIRE(benchmarks[0].owner == &ownerA);
    REQUIRE(benchmarks[0].factory().size() == 1);
    REQUIRE(benchmarks[0].factory()[0].variant == "owner-a");
    REQUIRE(benchmarks[1].owner == &ownerB);
    const auto otherBenchmarks = Registry::ListAvailableBenchmarks(otherBenchmarkType);
    REQUIRE(otherBenchmarks.size() == 1);
    REQUIRE(otherBenchmarks[0].owner == &ownerA);
    REQUIRE(Registry::ListAvailableBenchmarks("__ce_registry_test_missing_benchmark").empty());

    REQUIRE(Registry::UnregisterFlowgraph(flowgraphKey) == Result::SUCCESS);
    REQUIRE(Registry::ListAvailableFlowgraphs(flowgraphKey).empty());
    REQUIRE(Registry::UnregisterFlowgraph(flowgraphKey) == Result::ERROR);

    REQUIRE(Registry::UnregisterBenchmark(benchmarkType, &ownerA) == Result::SUCCESS);
    const auto remainingBenchmarks = Registry::ListAvailableBenchmarks(benchmarkType);
    REQUIRE(remainingBenchmarks.size() == 1);
    REQUIRE(remainingBenchmarks[0].owner == &ownerB);
    REQUIRE(Registry::ListAvailableBenchmarks(otherBenchmarkType).size() == 1);
    REQUIRE(Registry::UnregisterBenchmark(benchmarkType, &ownerA) == Result::ERROR);
}

TEST_CASE("Registry defers every registration kind until lookup", "[core][registry][deferred]") {
    PrepareRegistry();

    const std::string moduleType = "__ce_registry_test_deferred_module";
    const ProviderType provider = "__ce_registry_test_deferred_provider";
    const std::string blockType = "__ce_registry_test_deferred_all_block";
    const std::string flowgraphKey = "__ce_registry_test_deferred_all_flowgraph";
    const std::string benchmarkType = "__ce_registry_test_deferred_benchmark";
    const auto flowgraph = MakeFlowgraph(flowgraphKey, "Deferred Registry Flowgraph");
    auto expectedModule = MakeTestModule();
    auto expectedBlock = MakeTestBlock();
    int benchmarkOwner = 0;
    std::vector<std::string> order;
    RegistryCleanup cleanup;
    cleanup.module(moduleType, DeviceType::CPU, RuntimeType::NATIVE, provider);
    cleanup.block(blockType);
    cleanup.flowgraph(flowgraphKey);
    cleanup.benchmark(benchmarkType, &benchmarkOwner);

    REQUIRE(Registry::QueueStaticRegistration([&]() {
                order.emplace_back("module");
                return Registry::RegisterModule(
                    moduleType,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    provider,
                    [expectedModule](const auto&, const auto&) {
                        return expectedModule;
                    });
            }) == Result::SUCCESS);
    REQUIRE(Registry::QueueStaticRegistration([&]() {
                order.emplace_back("block");
                return Registry::RegisterBlock(blockType,
                                               "__ce_registry_test_domain",
                                               "Deferred Registry Block",
                                               "Deferred registry summary",
                                               "Deferred registry description",
                                               [expectedBlock]() {
                                                   return expectedBlock;
                                               });
            }) == Result::SUCCESS);
    REQUIRE(Registry::QueueStaticRegistration([&]() {
                order.emplace_back("flowgraph");
                return Registry::RegisterFlowgraph(flowgraphKey, flowgraph);
            }) == Result::SUCCESS);
    REQUIRE(Registry::QueueStaticRegistration([&]() {
                order.emplace_back("benchmark");
                return Registry::RegisterBenchmark(benchmarkType,
                                                   MakeBenchmarkFactory("deferred"),
                                                   &benchmarkOwner);
            }) == Result::SUCCESS);
    REQUIRE(order.empty());

    const auto modules = Registry::ListAvailableModules(moduleType);
    const std::vector<std::string> expectedOrder = {
        "module", "block", "flowgraph", "benchmark"};
    REQUIRE(order == expectedOrder);
    REQUIRE(modules.size() == 1);
    REQUIRE(Registry::ListAvailableBlocks(blockType).size() == 1);
    REQUIRE(Registry::ListAvailableFlowgraphs(flowgraphKey).size() == 1);
    const auto benchmarks = Registry::ListAvailableBenchmarks(benchmarkType);
    REQUIRE(benchmarks.size() == 1);
    REQUIRE(benchmarks[0].factory()[0].variant == "deferred");

    std::shared_ptr<Module> builtModule;
    REQUIRE(Registry::BuildModule(moduleType,
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  provider,
                                  builtModule) == Result::SUCCESS);
    REQUIRE(builtModule == expectedModule);

    std::shared_ptr<Block> builtBlock;
    REQUIRE(Registry::BuildBlock(blockType, builtBlock) == Result::SUCCESS);
    REQUIRE(builtBlock == expectedBlock);
}

TEST_CASE("Registry drains a test translation unit static registration once",
          "[core][registry][static]") {
    PrepareRegistry();

    const std::string benchmarkType = "__ce_registry_test_tu_static_benchmark";
    const auto registrations = Registry::ListAvailableBenchmarks(benchmarkType);
    REQUIRE(registrations.size() == 1);
    REQUIRE(registrations[0].owner != nullptr);
    REQUIRE(registrations[0].factory().size() == 1);
    REQUIRE(registrations[0].factory()[0].variant == "tu-static");

    RegistryCleanup cleanup;
    cleanup.benchmark(benchmarkType, registrations[0].owner);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    REQUIRE(Registry::ListAvailableBenchmarks(benchmarkType).size() == 1);
    REQUIRE(Registry::UnregisterBenchmark(benchmarkType,
                                          registrations[0].owner) == Result::SUCCESS);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    REQUIRE(Registry::ListAvailableBenchmarks(benchmarkType).empty());
}

TEST_CASE("Registry owns factory captures through registration snapshots", "[core][registry][ownership]") {
    PrepareRegistry();

    const std::string moduleType = "__ce_registry_test_owned_module_factory";
    const ProviderType provider = "__ce_registry_test_owned_provider";
    const std::string blockType = "__ce_registry_test_owned_block_factory";
    const std::string benchmarkType = "__ce_registry_test_owned_benchmark_factory";
    int benchmarkOwner = 0;
    RegistryCleanup cleanup;
    cleanup.module(moduleType, DeviceType::CPU, RuntimeType::NATIVE, provider);
    cleanup.block(blockType);
    cleanup.benchmark(benchmarkType, &benchmarkOwner);

    auto moduleCapture = std::make_shared<int>(1);
    std::weak_ptr<int> weakModuleCapture = moduleCapture;
    REQUIRE(Registry::RegisterModule(
                moduleType,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                provider,
                [moduleCapture](const auto&, const auto&) {
                    (void)moduleCapture;
                    return MakeTestModule();
                }) == Result::SUCCESS);
    moduleCapture.reset();
    REQUIRE_FALSE(weakModuleCapture.expired());

    auto moduleSnapshot = Registry::ListAvailableModules(moduleType);
    REQUIRE(Registry::UnregisterModule(moduleType,
                                       DeviceType::CPU,
                                       RuntimeType::NATIVE,
                                       provider) == Result::SUCCESS);
    REQUIRE_FALSE(weakModuleCapture.expired());
    moduleSnapshot.clear();
    REQUIRE(weakModuleCapture.expired());

    auto blockCapture = std::make_shared<int>(2);
    std::weak_ptr<int> weakBlockCapture = blockCapture;
    REQUIRE(Registry::RegisterBlock(blockType,
                                    "__ce_registry_test_domain",
                                    "Owned Block Factory",
                                    "Owned summary",
                                    "Owned description",
                                    [blockCapture]() {
                                        (void)blockCapture;
                                        return MakeTestBlock();
                                    }) == Result::SUCCESS);
    blockCapture.reset();
    REQUIRE_FALSE(weakBlockCapture.expired());

    auto blockSnapshot = Registry::ListAvailableBlocks(blockType);
    REQUIRE(Registry::UnregisterBlock(blockType) == Result::SUCCESS);
    REQUIRE_FALSE(weakBlockCapture.expired());
    blockSnapshot.clear();
    REQUIRE(weakBlockCapture.expired());

    auto benchmarkCapture = std::make_shared<int>(3);
    std::weak_ptr<int> weakBenchmarkCapture = benchmarkCapture;
    REQUIRE(Registry::RegisterBenchmark(
                benchmarkType,
                [benchmarkCapture]() {
                    Benchmark::Case benchmark;
                    benchmark.variant = std::to_string(*benchmarkCapture);
                    return std::vector<Benchmark::Case>{benchmark};
                },
                &benchmarkOwner) == Result::SUCCESS);
    benchmarkCapture.reset();
    REQUIRE_FALSE(weakBenchmarkCapture.expired());

    auto benchmarkSnapshot = Registry::ListAvailableBenchmarks(benchmarkType);
    REQUIRE(Registry::UnregisterBenchmark(benchmarkType, &benchmarkOwner) == Result::SUCCESS);
    REQUIRE_FALSE(weakBenchmarkCapture.expired());
    REQUIRE(benchmarkSnapshot[0].factory()[0].variant == "3");
    benchmarkSnapshot.clear();
    REQUIRE(weakBenchmarkCapture.expired());
}

TEST_CASE("Registry benchmark owners are non-owning snapshot identities",
          "[core][registry][ownership][snapshot]") {
    PrepareRegistry();

    const std::string benchmarkType = "__ce_registry_test_owner_snapshot";
    auto owner = std::make_shared<int>(7);
    std::weak_ptr<int> weakOwner = owner;
    const void* ownerIdentity = owner.get();
    RegistryCleanup cleanup;
    cleanup.benchmark(benchmarkType, ownerIdentity);

    REQUIRE(Registry::RegisterBenchmark(benchmarkType,
                                        MakeBenchmarkFactory("owner-snapshot"),
                                        ownerIdentity) == Result::SUCCESS);
    auto snapshot = Registry::ListAvailableBenchmarks(benchmarkType);
    REQUIRE(snapshot.size() == 1);
    REQUIRE(snapshot[0].owner == ownerIdentity);

    owner.reset();
    REQUIRE(weakOwner.expired());
    REQUIRE(snapshot[0].owner == ownerIdentity);
    REQUIRE(Registry::UnregisterBenchmark(benchmarkType, ownerIdentity) == Result::SUCCESS);
    REQUIRE(Registry::ListAvailableBenchmarks(benchmarkType).empty());
    REQUIRE(snapshot[0].factory()[0].variant == "owner-snapshot");
}

TEST_CASE("Registry keeps partial deferred effects and drops the failed batch",
          "[core][registry][deferred][errors]") {
    PrepareRegistry();

    const std::string flowgraphKey = "__ce_registry_test_partial_deferred_flowgraph";
    const auto flowgraph = MakeFlowgraph(flowgraphKey, "Partial Deferred Flowgraph");
    int completedCalls = 0;
    int failingCalls = 0;
    int skippedCalls = 0;
    auto completedCapture = std::make_shared<int>(1);
    auto failingCapture = std::make_shared<int>(2);
    auto skippedCapture = std::make_shared<int>(3);
    std::weak_ptr<int> weakCompletedCapture = completedCapture;
    std::weak_ptr<int> weakFailingCapture = failingCapture;
    std::weak_ptr<int> weakSkippedCapture = skippedCapture;
    RegistryCleanup cleanup;
    cleanup.flowgraph(flowgraphKey);

    REQUIRE(Registry::QueueStaticRegistration([&, completedCapture]() {
                (void)completedCapture;
                ++completedCalls;
                return Registry::RegisterFlowgraph(flowgraphKey, flowgraph);
            }) == Result::SUCCESS);
    REQUIRE(Registry::QueueStaticRegistration([&, failingCapture]() {
                (void)failingCapture;
                ++failingCalls;
                return Result::ERROR;
            }) == Result::SUCCESS);
    REQUIRE(Registry::QueueStaticRegistration([&, skippedCapture]() {
                (void)skippedCapture;
                ++skippedCalls;
                return Result::SUCCESS;
            }) == Result::SUCCESS);
    completedCapture.reset();
    failingCapture.reset();
    skippedCapture.reset();

    REQUIRE(Registry::DrainStaticRegistrations() == Result::ERROR);
    REQUIRE(completedCalls == 1);
    REQUIRE(failingCalls == 1);
    REQUIRE(skippedCalls == 0);
    REQUIRE(weakCompletedCapture.expired());
    REQUIRE(weakFailingCapture.expired());
    REQUIRE(weakSkippedCapture.expired());
    REQUIRE(Registry::ListAvailableFlowgraphs(flowgraphKey).size() == 1);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    REQUIRE(skippedCalls == 0);
}

TEST_CASE("Registry retains callbacks queued concurrently with a failed drain",
          "[core][registry][deferred][concurrency]") {
    PrepareRegistry();

    using namespace std::chrono_literals;
    std::promise<void> enteredPromise;
    auto entered = enteredPromise.get_future();
    std::promise<void> releasePromise;
    auto release = releasePromise.get_future().share();
    std::atomic<int> retainedCalls{0};
    Result drainResult = Result::SUCCESS;
    RegistryCleanup cleanup;

    REQUIRE(Registry::QueueStaticRegistration([&]() {
                enteredPromise.set_value();
                release.wait();
                return Result::SUCCESS;
            }) == Result::SUCCESS);
    REQUIRE(Registry::QueueStaticRegistration([]() {
                return Result::ERROR;
            }) == Result::SUCCESS);

    std::thread drainer([&]() {
        drainResult = Registry::DrainStaticRegistrations();
    });
    const bool drainEntered = entered.wait_for(5s) == std::future_status::ready;
    Result queueResult = Result::ERROR;
    if (drainEntered) {
        queueResult = Registry::QueueStaticRegistration([&]() {
            retainedCalls.fetch_add(1, std::memory_order_relaxed);
            return Result::SUCCESS;
        });
    }
    releasePromise.set_value();
    drainer.join();

    REQUIRE(drainEntered);
    REQUIRE(queueResult == Result::SUCCESS);
    REQUIRE(drainResult == Result::ERROR);
    REQUIRE(retainedCalls.load(std::memory_order_relaxed) == 0);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    REQUIRE(retainedCalls.load(std::memory_order_relaxed) == 1);
}

TEST_CASE("Registry drains nested deferred registrations in queue order", "[core][registry][deferred]") {
    PrepareRegistry();

    const std::string flowgraphKey = "__ce_registry_test_deferred_flowgraph";
    const std::string blockType = "__ce_registry_test_deferred_block";
    const auto flowgraph = MakeFlowgraph(flowgraphKey, "Deferred Flowgraph");
    std::vector<std::string> order;
    RegistryCleanup cleanup;
    cleanup.flowgraph(flowgraphKey);
    cleanup.block(blockType);

    REQUIRE(Registry::QueueStaticRegistration([&]() {
                order.emplace_back("outer-a");
                return Registry::QueueStaticRegistration([&]() {
                    order.emplace_back("nested");
                    return Registry::RegisterFlowgraph(flowgraphKey, flowgraph);
                });
            }) == Result::SUCCESS);
    REQUIRE(Registry::QueueStaticRegistration([&]() {
                order.emplace_back("outer-b");
                return Registry::RegisterBlock(blockType,
                                               "__ce_registry_test_domain",
                                               "Deferred Block",
                                               "Deferred summary",
                                               "Deferred description",
                                               []() {
                                                   return MakeTestBlock();
                                               });
            }) == Result::SUCCESS);

    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    const std::vector<std::string> expectedOrder = {"outer-a", "outer-b", "nested"};
    REQUIRE(order == expectedOrder);
    REQUIRE(Registry::ListAvailableFlowgraphs(flowgraphKey).size() == 1);
    REQUIRE(Registry::ListAvailableBlocks(blockType).size() == 1);

    bool discardedCallbackRan = false;
    auto discardedCapture = std::make_shared<int>(1);
    std::weak_ptr<int> weakDiscardedCapture = discardedCapture;
    REQUIRE(Registry::QueueStaticRegistration([&]() {
                discardedCallbackRan = true;
                return Result::SUCCESS;
            }) == Result::SUCCESS);
    REQUIRE(Registry::QueueStaticRegistration([discardedCapture]() {
                (void)discardedCapture;
                return Result::SUCCESS;
            }) == Result::SUCCESS);
    discardedCapture.reset();
    REQUIRE_FALSE(weakDiscardedCapture.expired());
    REQUIRE(Registry::DiscardStaticRegistrations() == Result::SUCCESS);
    REQUIRE(weakDiscardedCapture.expired());
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    REQUIRE_FALSE(discardedCallbackRan);
}

TEST_CASE("Registry contains deferred callback errors and exceptions", "[core][registry][errors]") {
    PrepareRegistry();
    RegistryCleanup cleanup;

    REQUIRE(Registry::QueueStaticRegistration({}) == Result::ERROR);

    bool nestedCallbackRan = false;
    Result nestedQueueResult = Result::ERROR;
    bool callbackAfterErrorRan = false;
    REQUIRE(Registry::QueueStaticRegistration([&]() {
                nestedQueueResult = Registry::QueueStaticRegistration([&]() {
                    nestedCallbackRan = true;
                    return Result::SUCCESS;
                });
                return Result::WARNING;
            }) == Result::SUCCESS);
    REQUIRE(Registry::QueueStaticRegistration([&]() {
                callbackAfterErrorRan = true;
                return Result::SUCCESS;
            }) == Result::SUCCESS);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::ERROR);
    REQUIRE(nestedQueueResult == Result::SUCCESS);
    REQUIRE_FALSE(nestedCallbackRan);
    REQUIRE_FALSE(callbackAfterErrorRan);

    bool recoveryCallbackRan = false;
    REQUIRE(Registry::QueueStaticRegistration([&]() {
                recoveryCallbackRan = true;
                return Result::SUCCESS;
            }) == Result::SUCCESS);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    REQUIRE(recoveryCallbackRan);
    REQUIRE_FALSE(nestedCallbackRan);
    REQUIRE_FALSE(callbackAfterErrorRan);

    REQUIRE(Registry::QueueStaticRegistration([]() -> Result {
                throw std::runtime_error("registry test exception");
            }) == Result::SUCCESS);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::ERROR);

    REQUIRE(Registry::QueueStaticRegistration([]() -> Result {
                throw Result::FATAL;
            }) == Result::SUCCESS);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::ERROR);

    REQUIRE(Registry::QueueStaticRegistration([]() -> Result {
                throw 7;
            }) == Result::SUCCESS);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::ERROR);

    REQUIRE(Registry::QueueStaticRegistration([]() {
                return Result::SUCCESS;
            }) == Result::SUCCESS);
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
}

TEST_CASE("Registry serializes concurrent registration list build unregister and drain",
          "[core][registry][concurrency]") {
    PrepareRegistry();

    constexpr int iterations = 64;
    const std::string anchorType = "__ce_registry_test_concurrent_anchor";
    const std::string churnType = "__ce_registry_test_concurrent_churn";
    const ProviderType provider = "__ce_registry_test_concurrent_provider";
    auto expectedModule = MakeTestModule();
    std::atomic<bool> concurrentFailure{false};
    std::atomic<int> deferredCalls{0};
    std::barrier<> startLine(5);
    std::barrier<> churnRegistered(2);
    std::barrier<> churnRemoved(2);
    RegistryCleanup cleanup;
    cleanup.module(anchorType, DeviceType::CPU, RuntimeType::NATIVE, provider);
    cleanup.module(churnType, DeviceType::CPU, RuntimeType::NATIVE, provider);

    REQUIRE(Registry::RegisterModule(
                anchorType,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                provider,
                [expectedModule](const auto&, const auto&) {
                    return expectedModule;
                }) == Result::SUCCESS);

    std::thread registrar([&]() {
        startLine.arrive_and_wait();
        for (int i = 0; i < iterations; ++i) {
            const auto result = Registry::RegisterModule(
                churnType,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                provider,
                [expectedModule](const auto&, const auto&) {
                    return expectedModule;
                });
            if (result != Result::SUCCESS) {
                concurrentFailure.store(true, std::memory_order_relaxed);
            }
            churnRegistered.arrive_and_wait();
            churnRemoved.arrive_and_wait();
        }
    });

    std::thread lister([&]() {
        startLine.arrive_and_wait();
        for (int i = 0; i < iterations; ++i) {
            const auto registrations = Registry::ListAvailableModules(anchorType);
            if (registrations.size() != 1 ||
                registrations[0].provider != provider ||
                !registrations[0].factory) {
                concurrentFailure.store(true, std::memory_order_relaxed);
            }
        }
    });

    std::thread builder([&]() {
        startLine.arrive_and_wait();
        for (int i = 0; i < iterations; ++i) {
            std::shared_ptr<Module> built;
            const auto result = Registry::BuildModule(anchorType,
                                                      DeviceType::CPU,
                                                      RuntimeType::NATIVE,
                                                      provider,
                                                      built);
            if (result != Result::SUCCESS || built != expectedModule) {
                concurrentFailure.store(true, std::memory_order_relaxed);
            }
        }
    });

    std::thread unregistrar([&]() {
        startLine.arrive_and_wait();
        for (int i = 0; i < iterations; ++i) {
            churnRegistered.arrive_and_wait();
            if (Registry::UnregisterModule(churnType,
                                           DeviceType::CPU,
                                           RuntimeType::NATIVE,
                                           provider) != Result::SUCCESS) {
                concurrentFailure.store(true, std::memory_order_relaxed);
            }
            churnRemoved.arrive_and_wait();
        }
    });

    std::thread deferredDrainer([&]() {
        startLine.arrive_and_wait();
        for (int i = 0; i < iterations; ++i) {
            if (Registry::QueueStaticRegistration([&]() {
                    deferredCalls.fetch_add(1, std::memory_order_relaxed);
                    return Result::SUCCESS;
                }) != Result::SUCCESS ||
                Registry::DrainStaticRegistrations() != Result::SUCCESS) {
                concurrentFailure.store(true, std::memory_order_relaxed);
            }
        }
    });

    registrar.join();
    lister.join();
    builder.join();
    unregistrar.join();
    deferredDrainer.join();

    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    REQUIRE_FALSE(concurrentFailure.load(std::memory_order_relaxed));
    REQUIRE(deferredCalls.load(std::memory_order_relaxed) == iterations);
    REQUIRE(Registry::ListAvailableModules(anchorType).size() == 1);
    REQUIRE(Registry::ListAvailableModules(churnType).empty());
}

// TODO: Probe factories that call Register/List/Build/Unregister, or drain a
// registering callback, once this test target exposes a timeout-controlled
// subprocess entry point. An in-process timeout cannot reclaim the worker
// blocked on the registry mutex, and fork would inherit process mutex state.

TEST_CASE("Registry factories can enqueue deferred registry work", "[core][registry][reentrant]") {
    PrepareRegistry();

    const std::string moduleType = "__ce_registry_test_reentrant_module";
    const ProviderType provider = "__ce_registry_test_reentrant_provider";
    const std::string blockType = "__ce_registry_test_reentrant_block";
    auto expectedModule = MakeTestModule();
    Result queueResult = Result::ERROR;
    int deferredCalls = 0;
    RegistryCleanup cleanup;
    cleanup.module(moduleType, DeviceType::CPU, RuntimeType::NATIVE, provider);
    cleanup.block(blockType);

    REQUIRE(Registry::RegisterModule(
                moduleType,
                DeviceType::CPU,
                RuntimeType::NATIVE,
                provider,
                [&](const auto&, const auto&) {
                    queueResult = Registry::QueueStaticRegistration([&]() {
                        ++deferredCalls;
                        return Registry::RegisterBlock(blockType,
                                                       "__ce_registry_test_domain",
                                                       "Reentrant Block",
                                                       "Reentrant summary",
                                                       "Reentrant description",
                                                       []() {
                                                           return MakeTestBlock();
                                                       });
                    });
                    return expectedModule;
                }) == Result::SUCCESS);

    std::shared_ptr<Module> builtModule;
    REQUIRE(Registry::BuildModule(moduleType,
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  provider,
                                  builtModule) == Result::SUCCESS);
    REQUIRE(builtModule == expectedModule);
    REQUIRE(queueResult == Result::SUCCESS);
    REQUIRE(deferredCalls == 0);

    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    REQUIRE(deferredCalls == 1);

    std::shared_ptr<Block> builtBlock;
    REQUIRE(Registry::BuildBlock(blockType, builtBlock) == Result::SUCCESS);
    REQUIRE(builtBlock != nullptr);
}
