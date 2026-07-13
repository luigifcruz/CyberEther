#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "jetstream/domains/visualization/waterfall/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/testing.hh"
#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct WaterfallImplAccess : Modules::WaterfallImpl {
    static auto ringStateMember() {
        return &WaterfallImplAccess::ringState;
    }

    static auto frequencyBinsMember() {
        return &WaterfallImplAccess::frequencyBins;
    }
};

const Modules::WaterfallRingState& RingState(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::WaterfallImpl>();
    if (!impl) {
        throw std::runtime_error("waterfall implementation is unavailable");
    }
    return impl->*WaterfallImplAccess::ringStateMember();
}

std::vector<F32> ReadFrequencyBins(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::WaterfallImpl>();
    if (!impl) {
        throw std::runtime_error("waterfall implementation is unavailable");
    }

    const Tensor& frequencyBins = impl->*WaterfallImplAccess::frequencyBinsMember();
    Tensor hostFrequencyBins;
    const Tensor* readableFrequencyBins = &frequencyBins;
    if (frequencyBins.device() != DeviceType::CPU) {
        if (hostFrequencyBins.create(DeviceType::CPU, frequencyBins) != Result::SUCCESS) {
            throw std::runtime_error("waterfall frequency bins are not host accessible");
        }
        readableFrequencyBins = &hostFrequencyBins;
    }

    const F32* data = readableFrequencyBins->data<F32>();
    return {data, data + readableFrequencyBins->size()};
}

void ApplyReferenceRows(std::vector<F32>& ring,
                        U64& writeIndex,
                        const Tensor& input,
                        const U64 height) {
    const U64 numberOfBatches = input.shape(0);
    const U64 numberOfElements = input.shape(1);
    const F32* inputData = input.data<F32>();
    for (U64 row = 0; row < numberOfBatches; ++row) {
        std::copy_n(inputData + row * numberOfElements,
                    numberOfElements,
                    ring.data() + ((writeIndex + row) % height) * numberOfElements);
    }
    writeIndex = (writeIndex + (numberOfBatches % height)) % height;
}

void ApplyReferenceWrite(std::vector<U64>& ring,
                         U64& writeIndex,
                         const std::vector<U64>& input,
                         const U64 height) {
    for (U64 row = 0; row < input.size(); ++row) {
        ring[(writeIndex + row) % height] = input[row];
    }
    writeIndex = (writeIndex + (input.size() % height)) % height;
}

void ApplyPlannedWrite(std::vector<U64>& ring,
                       Modules::WaterfallRingState& state,
                       const std::vector<U64>& input,
                       const U64 height) {
    const auto plan = Modules::PlanWaterfallWrite(state.writeIndex,
                                                  input.size(),
                                                  height);
    for (U64 row = 0; row < plan.rowCount; ++row) {
        ring[(plan.destinationRow + row) % height] = input[plan.sourceRow + row];
    }
    state.advance(input.size(), height);
}

}  // namespace

TEST_CASE("Waterfall ring retains newest rows for arbitrary batch counts",
          "[modules][waterfall][ring]") {
    constexpr U64 height = 5;
    const std::array<U64, 6> batchCounts = {1, height, height + 1, 2 * height,
                                            2 * height + 3, 3};

    std::vector<U64> reference(height, 0);
    std::vector<U64> planned(height, 0);
    Modules::WaterfallRingState state;
    U64 referenceWriteIndex = 0;
    U64 nextValue = 1;

    for (const U64 batchCount : batchCounts) {
        std::vector<U64> input(batchCount);
        std::iota(input.begin(), input.end(), nextValue);
        nextValue += batchCount;

        ApplyReferenceWrite(reference, referenceWriteIndex, input, height);
        ApplyPlannedWrite(planned, state, input, height);

        REQUIRE(planned == reference);
        REQUIRE(state.writeIndex == referenceWriteIndex);
    }

    std::vector<U64> chronological;
    for (U64 row = 0; row < height; ++row) {
        chronological.push_back(planned[(state.writeIndex + row) % height]);
    }
    REQUIRE(chronological == std::vector<U64>{nextValue - 5, nextValue - 4,
                                               nextValue - 3, nextValue - 2,
                                               nextValue - 1});
}

TEST_CASE("Waterfall dirty tracking covers full rotations and wraps",
          "[modules][waterfall][ring]") {
    constexpr U64 height = 5;
    Modules::WaterfallRingState state;

    state.advance(height, height);
    REQUIRE(state.writeIndex == 0);
    REQUIRE(state.dirtyRows == height);
    auto plan = state.dirtyPlan(height);
    REQUIRE(plan.startRow == 0);
    REQUIRE(plan.firstRowCount == height);
    REQUIRE(plan.secondRowCount == 0);

    state.clearDirty();
    state.advance(2 * height, height);
    REQUIRE(state.writeIndex == 0);
    REQUIRE(state.dirtyRows == height);

    state.clearDirty();
    state.advance(3, height);
    REQUIRE(state.writeIndex == 3);
    REQUIRE(state.dirtyRows == 3);
    plan = state.dirtyPlan(height);
    REQUIRE(plan.startRow == 0);
    REQUIRE(plan.firstRowCount == 3);
    REQUIRE(plan.secondRowCount == 0);

    state.clearDirty();
    state.advance(4, height);
    REQUIRE(state.writeIndex == 2);
    REQUIRE(state.dirtyRows == 4);
    plan = state.dirtyPlan(height);
    REQUIRE(plan.startRow == 3);
    REQUIRE(plan.firstRowCount == 2);
    REQUIRE(plan.secondRowCount == 2);

    state.clearDirty();
    state.advance(1, height);
    state.advance(4, height);
    REQUIRE(state.writeIndex == 2);
    REQUIRE(state.dirtyRows == height);
}

TEST_CASE("Waterfall preserves its ring cursor across runtime rebuilds",
          "[modules][waterfall][runtime]") {
    const auto implementations = Registry::ListAvailableModules("waterfall");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            constexpr U64 height = 5;
            constexpr U64 numberOfBatches = 2 * height + 2;

            Tensor cpuInput;
            REQUIRE(cpuInput.create(DeviceType::CPU,
                                    DataType::F32,
                                    {numberOfBatches, 3}) == Result::SUCCESS);
            std::iota(cpuInput.data<F32>(),
                      cpuInput.data<F32>() + cpuInput.size(),
                      1.0f);

            Tensor input;
            if (implementation.device == DeviceType::CPU) {
                input = cpuInput;
            } else {
                REQUIRE(input.create(implementation.device, cpuInput) == Result::SUCCESS);
            }

            TensorMap inputs;
            inputs["signal"].requested("source", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("waterfall",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::Waterfall config;
            config.height = height;
            REQUIRE(module->create("waterfall", config, inputs) == Result::SUCCESS);

            Runtime runtime("waterfall", implementation.device, implementation.runtime);
            REQUIRE(runtime.create({{"waterfall", module}}) == Result::SUCCESS);

            std::vector<F32> expectedFrequencyBins(height * 3, 0.0f);
            U64 expectedWriteIndex = 0;
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
            ApplyReferenceRows(expectedFrequencyBins, expectedWriteIndex, cpuInput, height);
            REQUIRE(RingState(module).writeIndex == expectedWriteIndex);
            REQUIRE(ReadFrequencyBins(module) == expectedFrequencyBins);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(RingState(module).writeIndex == expectedWriteIndex);

            REQUIRE(runtime.create({{"waterfall", module}}) == Result::SUCCESS);
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
            ApplyReferenceRows(expectedFrequencyBins, expectedWriteIndex, cpuInput, height);
            REQUIRE(RingState(module).writeIndex == expectedWriteIndex);
            REQUIRE(ReadFrequencyBins(module) == expectedFrequencyBins);
            REQUIRE(RingState(module).dirtyRows == height);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Waterfall module accepts valid F32 inputs", "[modules][waterfall]") {
    auto implementations = Registry::ListAvailableModules("waterfall");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("waterfall", impl.device, impl.runtime, impl.provider);

            Modules::Waterfall config;
            config.height = 32;
            config.interpolate = true;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {64}) ==
                    Result::SUCCESS);
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor batched;
            REQUIRE(batched.create(DeviceType::CPU, DataType::F32, {2, 64}) ==
                    Result::SUCCESS);
            ctx.setInput("signal", batched);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Waterfall module rejects invalid config and inputs",
          "[modules][waterfall][validation]") {
    auto implementations = Registry::ListAvailableModules("waterfall");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("height must be in range") {
                TestContext ctx("waterfall", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {32}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", input);

                Modules::Waterfall config;
                config.height = 0;
                ctx.setConfig(config);
                REQUIRE(ctx.run() == Result::ERROR);

                config.height = 2049;
                ctx.setConfig(config);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("dtype must be F32") {
                TestContext ctx("waterfall", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {32}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("rank must be one or two") {
                TestContext ctx("waterfall", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2, 2, 2}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }
        }
    }
}

TEST_CASE("Waterfall module supports repeated runs and config updates",
          "[modules][waterfall][state]") {
    auto implementations = Registry::ListAvailableModules("waterfall");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("waterfall", impl.device, impl.runtime, impl.provider);

            Modules::Waterfall config;
            config.height = 4;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2, 8}) ==
                    Result::SUCCESS);
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(ctx.run() == Result::SUCCESS);

            config.interpolate = false;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);

            config.height = 8;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}
