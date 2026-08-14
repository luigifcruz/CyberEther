#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "jetstream/domains/visualization/signal_view/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/testing.hh"

#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct SignalViewImplAccess : Modules::SignalViewImpl {
    static auto signalPointsMember() {
        return &SignalViewImplAccess::signalPoints;
    }

    static auto maxHoldPointsMember() {
        return &SignalViewImplAccess::maxHoldPoints;
    }

    static auto maxHoldWarmupBlocksMember() {
        return &SignalViewImplAccess::maxHoldWarmupBlocks;
    }

    static auto waterfallBinsMember() {
        return &SignalViewImplAccess::waterfallBins;
    }

    static auto waterfallHistoryMember() {
        return &SignalViewImplAccess::waterfallHistory;
    }
};

std::vector<F32> ReadTensor(const Tensor& tensor,
                            const char* name) {
    Tensor hostTensor;
    const Tensor* readable = &tensor;
    if (tensor.device() != DeviceType::CPU) {
        if (hostTensor.create(DeviceType::CPU, tensor) != Result::SUCCESS) {
            throw std::runtime_error(std::string(name) + " is not host accessible");
        }
        readable = &hostTensor;
    }

    const F32* data = readable->data<F32>();
    return {data, data + readable->size()};
}

std::vector<F32> ReadSignalPoints(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::SignalViewImpl>();
    if (!impl) {
        throw std::runtime_error(
            "signal view implementation is unavailable");
    }

    return ReadTensor(impl->*SignalViewImplAccess::signalPointsMember(),
                      "signal points");
}

void RequireSignalViewValidationError(const Registry::ModuleRegistration& impl,
                                      const Modules::SignalView& config,
                                      const Tensor& input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_view",
                                  impl.device,
                                  impl.runtime,
                                  impl.provider,
                                  module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
}

void RequireSignalViewValidationError(const Registry::ModuleRegistration& impl,
                                      const Modules::SignalView& config,
                                      const DataType dtype,
                                      const Shape& shape) {
    Tensor input;
    if (shape.empty()) {
        REQUIRE(input.create(impl.device, dtype, {1}) == Result::SUCCESS);
        REQUIRE(input.squeezeDims(0) == Result::SUCCESS);
    } else {
        REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    }
    RequireSignalViewValidationError(impl, config, input);
}

std::vector<F32> ReadMaxHoldPoints(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::SignalViewImpl>();
    if (!impl) {
        throw std::runtime_error(
            "signal view implementation is unavailable");
    }

    return ReadTensor(impl->*SignalViewImplAccess::maxHoldPointsMember(),
                      "max hold points");
}

std::vector<F32> ReadWaterfallBins(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::SignalViewImpl>();
    if (!impl) {
        throw std::runtime_error(
            "signal view implementation is unavailable");
    }

    return ReadTensor(impl->*SignalViewImplAccess::waterfallBinsMember(),
                      "waterfall history");
}

const Modules::WaterfallHistory&
ReadWaterfallHistory(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::SignalViewImpl>();
    if (!impl) {
        throw std::runtime_error(
            "signal view implementation is unavailable");
    }
    return impl->*SignalViewImplAccess::waterfallHistoryMember();
}

struct SignalViewSnapshot {
    std::vector<F32> signalPoints;
    std::vector<F32> waterfallBins;
};

SignalViewSnapshot ComputeSignalViewSnapshot(const Registry::ModuleRegistration& impl,
                                             const Tensor& cpuInput) {
    Tensor input;
    if (impl.device == DeviceType::CPU) {
        input = cpuInput;
    } else {
        REQUIRE(input.create(impl.device, cpuInput) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["signal"].requested("source", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_view",
                                  impl.device,
                                  impl.runtime,
                                  impl.provider,
                                  module) == Result::SUCCESS);

    Modules::SignalView config;
    config.mode = "lineplot_waterfall";
    config.fill = false;
    config.waterfallHeight = 2;
    REQUIRE(module->create("signal_view", config, inputs) == Result::SUCCESS);

    Runtime runtime("signal_view", impl.device, impl.runtime);
    REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);
    std::unordered_set<std::string> skipped;
    std::unordered_set<std::string> failed;
    REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

    SignalViewSnapshot snapshot{
        .signalPoints = ReadSignalPoints(module),
        .waterfallBins = ReadWaterfallBins(module),
    };
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
    return snapshot;
}

void ApplyReferenceRows(std::vector<F32>& ring,
                        U64& writeIndex,
                        const Tensor& input,
                        const U64 height) {
    const U64 batches = input.shape(0);
    const U64 width = input.shape(1);
    const U64 retained = std::min(batches, height);
    const U64 sourceRow = batches - retained;
    const U64 destinationRow = (writeIndex + (sourceRow % height)) % height;
    const F32* inputData = input.data<F32>();
    for (U64 row = 0; row < retained; ++row) {
        std::copy_n(inputData + (sourceRow + row) * width,
                    width,
                    ring.data() + ((destinationRow + row) % height) * width);
    }
    writeIndex = (writeIndex + (batches % height)) % height;
}

}  // namespace

TEST_CASE("Signal View module supports every visualization mode",
          "[modules][signal_view]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    const std::array<const char*, 3> modes = {
        "lineplot",
        "waterfall",
        "lineplot_waterfall",
    };

    for (const auto& implementation : implementations) {
        for (const char* mode : modes) {
            DYNAMIC_SECTION("Device: " << implementation.device
                            << " Runtime: " << implementation.runtime
                            << " Mode: " << mode) {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = mode;
                config.averaging = 4;
                config.decimation = 2;
                config.waterfallHeight = 32;
                config.xLabel = "Frequency";
                config.amplitudeLabel = "Power";
                config.waterfallLabel = "History";
                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2, 128}) ==
                        Result::SUCCESS);
                REQUIRE(SetSignalAxes(input, {
                    .sample = Index{1},
                    .batch = Index{0},
                }) == Result::SUCCESS);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::SUCCESS);

                Tensor channels;
                REQUIRE(channels.create(DeviceType::CPU, DataType::F32, {128}) ==
                        Result::SUCCESS);
                REQUIRE(SetSignalAxes(channels, {
                    .channel = Index{0},
                }) == Result::SUCCESS);
                ctx.setInput("signal", channels);
                REQUIRE(ctx.run() == Result::SUCCESS);
            }
        }
    }
}

TEST_CASE("Signal View validation is mode aware",
          "[modules][signal_view][validation]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {64}) ==
                    Result::SUCCESS);

            SECTION("invalid mode") {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = "invalid";
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("line settings") {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = "lineplot";
                config.decimation = 0;
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);

                config.decimation = 1;
                config.averaging = 0;
                ctx.setConfig(config);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("waterfall settings") {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = "waterfall";
                config.waterfallHeight = 0;
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);

                config.waterfallHeight = 8193;
                ctx.setConfig(config);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("waterfall accepts one column") {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = "waterfall";
                ctx.setConfig(config);
                Tensor oneColumn;
                REQUIRE(oneColumn.create(DeviceType::CPU, DataType::F32, {1}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", oneColumn);
                REQUIRE(ctx.run() == Result::SUCCESS);
            }
        }
    }
}

TEST_CASE("Signal View rejects unsupported input tensors",
          "[modules][signal_view][validation]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             DataType::CF32,
                                             {64});
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             DataType::F32,
                                             {});
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             DataType::F32,
                                             {2, 2, 2});

            Modules::SignalView decimated;
            decimated.decimation = 2;
            RequireSignalViewValidationError(implementation,
                                             decimated,
                                             DataType::F32,
                                             {3});
        }
    }
}

TEST_CASE("Signal View validates signal axis roles",
          "[modules][signal_view][validation][metadata]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor missing(implementation.device, DataType::F32, {2, 32});
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             missing);

            Tensor malformed(implementation.device, DataType::F32, {32});
            REQUIRE(malformed.setAttribute(std::string(SampleAxisAttribute),
                                           I64{0}) == Result::SUCCESS);
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             malformed);

            Tensor mixed(implementation.device, DataType::F32, {2, 32});
            REQUIRE(SetSignalAxes(mixed, {
                .sample = Index{1},
                .channel = Index{0},
            }) == Result::SUCCESS);
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             mixed);

            Tensor auxiliary(implementation.device, DataType::F32, {2, 32});
            REQUIRE(SetSignalAxes(auxiliary, {
                .sample = Index{1},
            }) == Result::SUCCESS);
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             auxiliary);
        }
    }
}

TEST_CASE("Signal View rejects non-finite configuration and metadata",
          "[modules][signal_view][validation]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {64}) ==
                    Result::SUCCESS);

            Modules::SignalView config;
            config.rangeMin = -std::numeric_limits<F32>::infinity();
            RequireSignalViewValidationError(implementation, config, input);

            REQUIRE(input.setAttribute("sampleRate", F64{1.0e6}) ==
                    Result::SUCCESS);
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             input);
        }
    }
}

TEST_CASE("Waterfall history tracks wrapped dirty rows",
          "[modules][signal_view][waterfall]") {
    constexpr U64 height = 5;
    Modules::WaterfallHistory history;

    history.advance(height, height);
    REQUIRE(history.writeIndex == 0);
    REQUIRE(history.dirtyRows == height);
    auto dirty = history.dirtyPlan(height);
    REQUIRE(dirty.startRow == 0);
    REQUIRE(dirty.firstRowCount == height);
    REQUIRE(dirty.secondRowCount == 0);

    history.clearDirty();
    history.advance(3, height);
    history.advance(4, height);
    REQUIRE(history.writeIndex == 2);
    REQUIRE(history.dirtyRows == height);
    dirty = history.dirtyPlan(height);
    REQUIRE(dirty.firstRowCount + dirty.secondRowCount == height);
}

TEST_CASE("Combined signal view keeps full-resolution waterfall rows",
          "[modules][signal_view][waterfall][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            constexpr U64 height = 3;
            constexpr U64 batches = 5;
            constexpr U64 rowWidth = 6;

            Tensor cpuInput;
            REQUIRE(cpuInput.create(DeviceType::CPU, DataType::F32,
                                    {batches, rowWidth}) == Result::SUCCESS);
            REQUIRE(SetSignalAxes(cpuInput, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::iota(cpuInput.data<F32>(),
                      cpuInput.data<F32>() + cpuInput.size(),
                      1.0f);

            Tensor input;
            if (implementation.device == DeviceType::CPU) {
                input = cpuInput;
            } else {
                REQUIRE(input.create(implementation.device, cpuInput) ==
                        Result::SUCCESS);
            }

            TensorMap inputs;
            inputs["signal"].requested("source", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot_waterfall";
            config.decimation = 2;
            config.waterfallHeight = height;
            REQUIRE(module->create("signal_view", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            const std::vector<F32> expected = {
                19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
            };
            REQUIRE(ReadWaterfallBins(module) == expected);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == 2);
            REQUIRE(ReadWaterfallHistory(module).dirtyRows == height);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View indexes sample and channel batch layouts equivalently",
           "[modules][signal_view][layout]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            for (const bool useChannelAxis : {false, true}) {
                DYNAMIC_SECTION("Element axis: "
                                << (useChannelAxis ? "channel" : "sample")) {
                    Tensor leading(DeviceType::CPU, DataType::F32, {2, 3});
                    const F32 leadingData[] = {
                        0.25f, 0.50f, 0.75f,
                        0.25f, 0.75f, 0.50f,
                    };
                    std::copy(std::begin(leadingData), std::end(leadingData),
                              leading.data<F32>());
                    SignalAxes leadingAxes{.batch = Index{0}};
                    if (useChannelAxis) {
                        leadingAxes.channel = Index{1};
                    } else {
                        leadingAxes.sample = Index{1};
                    }
                    REQUIRE(SetSignalAxes(leading, leadingAxes) == Result::SUCCESS);

                    Tensor trailing(DeviceType::CPU, DataType::F32, {3, 2});
                    const F32 trailingData[] = {
                        0.25f, 0.25f,
                        0.50f, 0.75f,
                        0.75f, 0.50f,
                    };
                    std::copy(std::begin(trailingData), std::end(trailingData),
                              trailing.data<F32>());
                    SignalAxes trailingAxes{.batch = Index{1}};
                    if (useChannelAxis) {
                        trailingAxes.channel = Index{0};
                    } else {
                        trailingAxes.sample = Index{0};
                    }
                    REQUIRE(SetSignalAxes(trailing, trailingAxes) == Result::SUCCESS);

                    const auto leadingSnapshot =
                        ComputeSignalViewSnapshot(implementation, leading);
                    const auto trailingSnapshot =
                        ComputeSignalViewSnapshot(implementation, trailing);
                    REQUIRE(trailingSnapshot.signalPoints ==
                            leadingSnapshot.signalPoints);
                    REQUIRE(trailingSnapshot.waterfallBins ==
                            leadingSnapshot.waterfallBins);
                }
            }
        }
    }
}

TEST_CASE("Signal View preserves waterfall history across runtime rebuilds",
          "[modules][signal_view][waterfall][runtime]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            constexpr U64 height = 5;
            constexpr U64 batches = 2 * height + 2;
            constexpr U64 width = 3;

            Tensor cpuInput(DeviceType::CPU, DataType::F32, {batches, width});
            REQUIRE(SetSignalAxes(cpuInput, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::iota(cpuInput.data<F32>(),
                      cpuInput.data<F32>() + cpuInput.size(),
                      1.0f);

            Tensor input;
            if (implementation.device == DeviceType::CPU) {
                input = cpuInput;
            } else {
                REQUIRE(input.create(implementation.device, cpuInput) ==
                        Result::SUCCESS);
            }

            TensorMap inputs;
            inputs["signal"].requested("source", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            Modules::SignalView config;
            config.mode = "waterfall";
            config.waterfallHeight = height;
            REQUIRE(module->create("signal_view", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::vector<F32> expected(height * width, 0.0f);
            U64 expectedWriteIndex = 0;
            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            ApplyReferenceRows(expected, expectedWriteIndex, cpuInput, height);
            REQUIRE(ReadWaterfallBins(module) == expected);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == expectedWriteIndex);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            ApplyReferenceRows(expected, expectedWriteIndex, cpuInput, height);
            REQUIRE(ReadWaterfallBins(module) == expected);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == expectedWriteIndex);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View reconfigure preserves applied waterfall state",
          "[modules][signal_view][waterfall][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {2, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            Modules::SignalView config;
            config.mode = "waterfall";
            config.waterfallHeight = 4;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            Parser::Map recreate;
            recreate["waterfallHeight"] = U64{8};
            REQUIRE(module->reconfigure(recreate, true) == Result::SUCCESS);
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            REQUIRE(module->reconfigure(recreate) == Result::RECREATE);
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            const auto& applied =
                static_cast<const Modules::SignalView&>(module->config());
            REQUIRE(applied.waterfallHeight == config.waterfallHeight);

            Parser::Map rejected;
            rejected["waterfallHeight"] = U64{0};
            REQUIRE(module->reconfigure(rejected) == Result::ERROR);
            REQUIRE(module->state() == Module::State::CREATED);
            REQUIRE(applied.waterfallHeight == config.waterfallHeight);
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View configuration omits fixed rendering settings",
          "[modules][signal_view][config]") {
    Modules::SignalView config;
    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE_FALSE(serialized.contains("interpolate"));
    REQUIRE_FALSE(serialized.contains("waterfallInterpolate"));
    REQUIRE_FALSE(serialized.contains("thickness"));
}

TEST_CASE("Signal View serializes plot labels", "[modules][signal_view][config]") {
    Modules::SignalView config;
    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE(std::any_cast<bool>(serialized.at("fill")));
    REQUIRE(std::any_cast<std::string>(serialized.at("xLabel")) ==
            "Frequency (MHz)");
    REQUIRE(std::any_cast<std::string>(serialized.at("amplitudeLabel")) ==
            "Amplitude (dBFS)");
    REQUIRE(std::any_cast<std::string>(serialized.at("waterfallLabel")) == "Time");

    config.xLabel = "";
    config.fill = false;
    config.amplitudeLabel = "Power";
    config.waterfallLabel = "History";
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE_FALSE(std::any_cast<bool>(serialized.at("fill")));
    REQUIRE(serialized.contains("xLabel"));
    REQUIRE(std::any_cast<std::string>(serialized.at("xLabel")).empty());
    REQUIRE(std::any_cast<std::string>(serialized.at("amplitudeLabel")) == "Power");
    REQUIRE(std::any_cast<std::string>(serialized.at("waterfallLabel")) == "History");
}

TEST_CASE("Lineplot helpers preserve indexing and max-hold warmup",
           "[modules][signal_view][lineplot][regression]") {
    REQUIRE(Modules::detail::LineplotInputIndex(1, 1, 5, 1, 2) == 7);
    REQUIRE_FALSE(Modules::detail::LineplotMaxHoldReady(2, 4));
    REQUIRE(Modules::detail::LineplotMaxHoldReady(3, 4));
    REQUIRE(Modules::detail::LineplotMaxHoldReady(0, 1));

    std::array<F32, 6> signalPoints;
    std::array<F32, 6> maxHoldPoints;
    Modules::detail::InitializeLineplotPoints(signalPoints.data(),
                                              maxHoldPoints.data(), 3);
    REQUIRE(signalPoints == std::array<F32, 6>{-1.0f, 0.0f, 0.0f, 0.0f,
                                               1.0f, 0.0f});
    REQUIRE(maxHoldPoints == std::array<F32, 6>{-1.0f, -1.0f, 0.0f, -1.0f,
                                                1.0f, -1.0f});
}

TEST_CASE("Signal View clamps amplitudes before averaging",
          "[modules][signal_view][lineplot][averaging][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor cpuInput(DeviceType::CPU, DataType::F32, {2, 4});
            REQUIRE(SetSignalAxes(cpuInput, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(cpuInput.data<F32>(), cpuInput.size(),
                        -std::numeric_limits<F32>::infinity());

            Tensor input;
            if (implementation.device == DeviceType::CPU) {
                input = cpuInput;
            } else {
                REQUIRE(input.create(implementation.device, cpuInput) ==
                        Result::SUCCESS);
            }

            TensorMap inputs;
            inputs["signal"].requested("source", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 2;
            config.fill = false;
            REQUIRE(module->create("signal_view", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            const auto firstPoints = ReadSignalPoints(module);
            for (const F32 point : firstPoints) {
                REQUIRE(std::isfinite(point));
            }

            std::fill_n(cpuInput.data<F32>(), cpuInput.size(), 2.0f);
            if (implementation.device != DeviceType::CPU) {
                REQUIRE(input.copyFrom(cpuInput) == Result::SUCCESS);
            }
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            const auto recoveredPoints = ReadSignalPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(std::isfinite(recoveredPoints[(index * 2) + 1]));
                REQUIRE(recoveredPoints[(index * 2) + 1] <= 1.0f);
                REQUIRE(recoveredPoints[(index * 2) + 1] >
                        firstPoints[(index * 2) + 1]);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View clears lineplot history on range reconfiguration",
          "[modules][signal_view][lineplot][reconfigure][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {1, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.25f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 4;
            config.maxHold = true;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 8; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }

            const auto settledHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(settledHold[(index * 2) + 1] > -1.0f);
            }

            Parser::Map range;
            range["rangeMin"] = F32{-50.0f};
            range["rangeMax"] = F32{50.0f};
            REQUIRE(module->reconfigure(range) == Result::SUCCESS);

            const auto* impl = module->getImpl<Modules::SignalViewImpl>();
            REQUIRE(impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember() == 0);
            const auto resetHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(resetHold[(index * 2) + 1] == -1.0f);
            }

            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            const auto freshPoints = ReadSignalPoints(module);
            const auto freshHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(freshPoints[(index * 2) + 1] == -0.125f);
                REQUIRE(freshHold[(index * 2) + 1] == -1.0f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View clears lineplot history on averaging reconfiguration",
          "[modules][signal_view][lineplot][reconfigure][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {1, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.25f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 4;
            config.maxHold = true;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 8; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }

            Parser::Map averaging;
            averaging["averaging"] = U64{16};
            REQUIRE(module->reconfigure(averaging) == Result::SUCCESS);

            const auto* impl = module->getImpl<Modules::SignalViewImpl>();
            REQUIRE(impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember() == 0);
            const auto resetHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(resetHold[(index * 2) + 1] == -1.0f);
            }

            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            const auto freshPoints = ReadSignalPoints(module);
            const auto freshHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(freshPoints[(index * 2) + 1] == -0.03125f);
                REQUIRE(freshHold[(index * 2) + 1] == -1.0f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View preserves waterfall history on averaging reconfiguration",
          "[modules][signal_view][waterfall][reconfigure][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {2, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.5f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot_waterfall";
            config.averaging = 4;
            config.maxHold = true;
            config.waterfallHeight = 4;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 3; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }
            REQUIRE(ReadWaterfallHistory(module).writeIndex == 2);

            Parser::Map averaging;
            averaging["averaging"] = U64{16};
            REQUIRE(module->reconfigure(averaging) == Result::SUCCESS);

            const auto* impl = module->getImpl<Modules::SignalViewImpl>();
            REQUIRE(impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember() == 0);
            const auto& history = ReadWaterfallHistory(module);
            REQUIRE(history.writeIndex == 2);

            const auto preservedBins = ReadWaterfallBins(module);
            for (const F32 value : preservedBins) {
                REQUIRE(value == 0.5f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View clears waterfall history on range reconfiguration",
          "[modules][signal_view][waterfall][reconfigure][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {2, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.5f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "waterfall";
            config.waterfallHeight = 4;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 3; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }
            REQUIRE(ReadWaterfallHistory(module).writeIndex == 2);

            Parser::Map range;
            range["rangeMin"] = F32{-50.0f};
            range["rangeMax"] = F32{50.0f};
            REQUIRE(module->reconfigure(range) == Result::SUCCESS);

            const auto& history = ReadWaterfallHistory(module);
            REQUIRE(history.writeIndex == 0);
            REQUIRE(history.dirtyRows == 4);
            const auto resetBins = ReadWaterfallBins(module);
            for (const F32 value : resetBins) {
                REQUIRE(value == 0.0f);
            }

            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            const auto freshBins = ReadWaterfallBins(module);
            for (U64 row = 0; row < 4; ++row) {
                for (U64 sample = 0; sample < 8; ++sample) {
                    REQUIRE(freshBins[(row * 8) + sample] ==
                            (row < 2 ? 0.5f : 0.0f));
                }
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View max hold captures the first observation with averaging one",
          "[modules][signal_view][lineplot][maxhold][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {1, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.25f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 1;
            config.maxHold = true;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            const auto hold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(hold[(index * 2) + 1] == -0.5f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View max hold captures the first fully averaged observation",
          "[modules][signal_view][lineplot][maxhold][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {1, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.25f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 4;
            config.maxHold = true;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 4; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }

            const auto hold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(hold[(index * 2) + 1] == -0.341796875f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
