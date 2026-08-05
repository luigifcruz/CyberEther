#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "jetstream/domains/visualization/lineplot/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/testing.hh"

#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct LineplotImplAccess : Modules::LineplotImpl {
    static auto signalPointsMember() {
        return &LineplotImplAccess::signalPoints;
    }
};

std::vector<F32> ReadSignalPoints(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::LineplotImpl>();
    if (!impl) {
        throw std::runtime_error("lineplot implementation is unavailable");
    }

    const Tensor& signalPoints = impl->*LineplotImplAccess::signalPointsMember();
    Tensor hostSignalPoints;
    const Tensor* readableSignalPoints = &signalPoints;
    if (signalPoints.device() != DeviceType::CPU) {
        if (hostSignalPoints.create(DeviceType::CPU, signalPoints) != Result::SUCCESS) {
            throw std::runtime_error("lineplot signal points are not host accessible");
        }
        readableSignalPoints = &hostSignalPoints;
    }

    const F32* data = readableSignalPoints->data<F32>();
    return {data, data + readableSignalPoints->size()};
}

void SetDefaultSignalAxes(Tensor& input) {
    if (input.rank() <= 1) {
        return;
    }

    SignalAxes axes{.sample = input.rank() - 1};
    if (input.rank() > 1) {
        axes.batch = 0;
    }
    REQUIRE(SetSignalAxes(input, axes) == Result::SUCCESS);
}

std::vector<F32> ComputeLineplotPoints(
    const Registry::ModuleRegistration& implementation,
    const Tensor& cpuInput) {
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
    REQUIRE(Registry::BuildModule("lineplot", implementation.device,
                                  implementation.runtime, implementation.provider,
                                  module) == Result::SUCCESS);

    Modules::Lineplot config;
    config.averaging = 1;
    REQUIRE(module->create("lineplot", config, inputs) == Result::SUCCESS);

    Runtime runtime("lineplot", implementation.device, implementation.runtime);
    REQUIRE(runtime.create({{"lineplot", module}}) == Result::SUCCESS);
    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

    auto points = ReadSignalPoints(module);
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
    return points;
}

void RequireLineplotValidationError(const Registry::ModuleRegistration& impl,
                                    const Modules::Lineplot& config,
                                    const Tensor& input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("lineplot", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
}

void RequireLineplotValidationError(const Registry::ModuleRegistration& impl,
                                    const Modules::Lineplot& config,
                                    const DataType dtype,
                                    const Shape& shape,
                                    const bool broadcast = false) {
    Tensor input;
    if (broadcast) {
        REQUIRE(input.create(impl.device, dtype, Shape(shape.size(), 1)) ==
                Result::SUCCESS);
        REQUIRE(input.broadcastTo(shape) == Result::SUCCESS);
    } else if (shape.empty()) {
        REQUIRE(input.create(impl.device, dtype, {1}) == Result::SUCCESS);
        REQUIRE(input.squeezeDims(0) == Result::SUCCESS);
    } else {
        REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    }

    SetDefaultSignalAxes(input);

    RequireLineplotValidationError(impl, config, input);
}

}  // namespace

TEST_CASE("Lineplot module accepts valid F32 inputs", "[modules][lineplot]") {
    auto implementations = Registry::ListAvailableModules("lineplot");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);

            Modules::Lineplot config;
            config.averaging = 4;
            config.decimation = 2;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {128}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {.sample = Index{0}}) == Result::SUCCESS);
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor batched;
            REQUIRE(batched.create(DeviceType::CPU, DataType::F32, {2, 128}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(batched, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            ctx.setInput("signal", batched);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor channels;
            REQUIRE(channels.create(DeviceType::CPU, DataType::F32, {128}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(channels, {.channel = Index{0}}) ==
                    Result::SUCCESS);
            ctx.setInput("signal", channels);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Lineplot module rejects invalid configuration values",
          "[modules][lineplot][validation]") {
    auto implementations = Registry::ListAvailableModules("lineplot");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("averaging must be positive") {
                Modules::Lineplot config;
                config.averaging = 0;
                RequireLineplotValidationError(impl, config, DataType::F32, {64});
            }

            SECTION("decimation must be positive") {
                Modules::Lineplot config;
                config.decimation = 0;
                RequireLineplotValidationError(impl, config, DataType::F32, {64});
            }

            SECTION("grid dimensions must be at least two") {
                Modules::Lineplot config;
                config.numberOfVerticalLines = 1;
                RequireLineplotValidationError(impl, config, DataType::F32, {64});

                config.numberOfVerticalLines = 11;
                config.numberOfHorizontalLines = 1;
                RequireLineplotValidationError(impl, config, DataType::F32, {64});
            }

            SECTION("thickness must be finite and positive") {
                for (const F32 thickness : {
                         0.0f,
                         -1.0f,
                         std::numeric_limits<F32>::quiet_NaN(),
                         std::numeric_limits<F32>::infinity(),
                         -std::numeric_limits<F32>::infinity(),
                     }) {
                    Modules::Lineplot config;
                    config.thickness = thickness;
                    RequireLineplotValidationError(impl, config,
                                                   DataType::F32, {64});
                }
            }
        }
    }
}

TEST_CASE("Lineplot module rejects invalid input dtype and shape",
          "[modules][lineplot][validation]") {
    auto implementations = Registry::ListAvailableModules("lineplot");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("dtype must be F32") {
                RequireLineplotValidationError(impl, Modules::Lineplot{},
                                               DataType::CF32, {64});
            }

            SECTION("rank must be one or two") {
                RequireLineplotValidationError(impl, Modules::Lineplot{},
                                               DataType::F32, {});
                RequireLineplotValidationError(impl, Modules::Lineplot{},
                                               DataType::F32, {2, 2, 2});
            }

            SECTION("multi-axis signal roles must be present and well formed") {
                Tensor missing(impl.device, DataType::F32, {2, 64});
                RequireLineplotValidationError(impl, Modules::Lineplot{}, missing);

                Tensor malformed(impl.device, DataType::F32, {64});
                REQUIRE(malformed.setAttribute(std::string(SampleAxisAttribute),
                                               I64{0}) == Result::SUCCESS);
                RequireLineplotValidationError(impl, Modules::Lineplot{}, malformed);
            }

            SECTION("mixed signal roles and auxiliary dimensions are unsupported") {
                Tensor mixed(impl.device, DataType::F32, {2, 64});
                REQUIRE(SetSignalAxes(mixed, {
                    .sample = Index{1},
                    .channel = Index{0},
                }) == Result::SUCCESS);
                RequireLineplotValidationError(impl, Modules::Lineplot{}, mixed);

                Tensor auxiliary(impl.device, DataType::F32, {2, 64});
                REQUIRE(SetSignalAxes(auxiliary, {.sample = Index{1}}) ==
                        Result::SUCCESS);
                RequireLineplotValidationError(impl, Modules::Lineplot{}, auxiliary);
            }

            SECTION("effective number of elements must be at least two") {
                Modules::Lineplot config;
                config.decimation = 2;
                RequireLineplotValidationError(impl, config,
                                               DataType::F32, {3});
            }
        }
    }
}

TEST_CASE("Lineplot module rejects malformed optional metadata during validation",
          "[modules][lineplot][validation][metadata]") {
    const auto implementations = Registry::ListAvailableModules("lineplot");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            for (const std::string& key : {std::string("frequency"),
                                           std::string("sampleRate")}) {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::F32, {64}) ==
                        Result::SUCCESS);
                REQUIRE(SetSignalAxes(input, {.sample = Index{0}}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute(key, F64{1000000.0}) ==
                        Result::SUCCESS);
                RequireLineplotValidationError(impl, Modules::Lineplot{}, input);
            }
        }
    }
}

TEST_CASE("Lineplot module rejects unsupported rendering geometry during validation",
          "[modules][lineplot][validation][size]") {
    const auto implementations = Registry::ListAvailableModules("lineplot");
    REQUIRE(!implementations.empty());

    const U64 maxRenderScalarCount = std::min(
        static_cast<U64>(std::numeric_limits<std::size_t>::max()) / sizeof(F32),
        static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(F32));
    const U64 maxElementCount = std::min({
        static_cast<U64>(std::numeric_limits<U32>::max()),
        static_cast<U64>(std::numeric_limits<U32>::max()) / 4 + 1,
        maxRenderScalarCount / 2,
        maxRenderScalarCount / 16 + 1,
    });

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireLineplotValidationError(impl, Modules::Lineplot{},
                                           DataType::F32,
                                           {maxElementCount + 1}, true);

            Modules::Lineplot gridConfig;
            gridConfig.numberOfVerticalLines = std::numeric_limits<U32>::max();
            RequireLineplotValidationError(impl, gridConfig,
                                           DataType::F32, {64});
        }
    }
}

TEST_CASE("Lineplot module handles repeated computes and config updates",
          "[modules][lineplot][state]") {
    auto implementations = Registry::ListAvailableModules("lineplot");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {64}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {.sample = Index{0}}) == Result::SUCCESS);
            ctx.setInput("signal", input);
            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);

            Modules::Lineplot config;
            config.averaging = 8;
            REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);

            config.decimation = 2;
            REQUIRE(ctx.reconfigure(config) == Result::RECREATE);
            REQUIRE(ctx.stop() == Result::SUCCESS);

            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Lineplot clamps amplitudes before averaging",
          "[modules][lineplot][averaging][nonfinite][regression]") {
    const auto implementations = Registry::ListAvailableModules("lineplot");

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor cpuInput(DeviceType::CPU, DataType::F32, {2, 4});
            REQUIRE(SetSignalAxes(cpuInput, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(cpuInput.data<F32>(),
                        cpuInput.size(),
                        -std::numeric_limits<F32>::infinity());

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
            REQUIRE(Registry::BuildModule("lineplot",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::Lineplot config;
            config.averaging = 2;
            REQUIRE(module->create("lineplot", config, inputs) == Result::SUCCESS);

            Runtime runtime("lineplot", implementation.device, implementation.runtime);
            REQUIRE(runtime.create({{"lineplot", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

            const auto firstPoints = ReadSignalPoints(module);
            for (U64 index = 0; index < firstPoints.size(); ++index) {
                REQUIRE(std::isfinite(firstPoints[index]));
            }

            std::fill_n(cpuInput.data<F32>(), cpuInput.size(), 1.0f);
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

            const auto recoveredPoints = ReadSignalPoints(module);
            for (U64 index = 0; index < recoveredPoints.size(); ++index) {
                REQUIRE(std::isfinite(recoveredPoints[index]));
            }
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(recoveredPoints[(index * 2) + 1] >
                        firstPoints[(index * 2) + 1]);
            }

            std::fill_n(cpuInput.data<F32>(), cpuInput.size(), 2.0f);
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

            const auto finiteOutOfRangePoints = ReadSignalPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(finiteOutOfRangePoints[(index * 2) + 1] <= 1.0f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Lineplot batched decimation uses the original row width",
          "[modules][lineplot][decimation][regression]") {
    const Shape shape = {2, 5};
    const U64 decimation = 2;
    const U64 numberOfElements = shape[1] / decimation;
    const F32 input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    F32 sums[] = {0.0f, 0.0f};

    for (U64 batch = 0; batch < shape[0]; ++batch) {
        for (U64 index = 0; index < numberOfElements; ++index) {
            sums[index] += input[Modules::detail::LineplotInputIndex(
                batch, index, shape[1], 1, decimation)];
        }
    }

    REQUIRE(sums[0] == 11.0f);
    REQUIRE(sums[1] == 33.0f);
}

TEST_CASE("Lineplot indexes sample and channel batch layouts equivalently",
           "[modules][lineplot][layout]") {
    const auto implementations = Registry::ListAvailableModules("lineplot");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            for (const bool useChannelAxis : {false, true}) {
                DYNAMIC_SECTION("Element axis: "
                                << (useChannelAxis ? "channel" : "sample")) {
                    Tensor leading(DeviceType::CPU, DataType::F32, {2, 4});
                    const F32 leadingData[] = {
                        0.1f, 0.2f, 0.3f, 0.4f,
                        0.2f, 0.2f, 0.2f, 0.2f,
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

                    Tensor trailing(DeviceType::CPU, DataType::F32, {4, 2});
                    const F32 trailingData[] = {
                        0.1f, 0.2f,
                        0.2f, 0.2f,
                        0.3f, 0.2f,
                        0.4f, 0.2f,
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

                    REQUIRE(ComputeLineplotPoints(implementation, trailing) ==
                            ComputeLineplotPoints(implementation, leading));
                }
            }
        }
    }
}
