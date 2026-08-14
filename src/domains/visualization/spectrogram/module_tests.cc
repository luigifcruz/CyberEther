#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "jetstream/domains/visualization/spectrogram/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/testing.hh"

#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct SpectrogramImplAccess : Modules::SpectrogramImpl {
    static auto frequencyBinsMember() {
        return &SpectrogramImplAccess::frequencyBins;
    }
};

std::vector<F32> ReadFrequencyBins(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::SpectrogramImpl>();
    if (!impl) {
        throw std::runtime_error("spectrogram implementation is unavailable");
    }

    const Tensor& frequencyBins = impl->*SpectrogramImplAccess::frequencyBinsMember();
    Tensor hostFrequencyBins;
    const Tensor* readableFrequencyBins = &frequencyBins;
    if (frequencyBins.device() != DeviceType::CPU) {
        if (hostFrequencyBins.create(DeviceType::CPU, frequencyBins) != Result::SUCCESS) {
            throw std::runtime_error("spectrogram frequency bins are not host accessible");
        }
        readableFrequencyBins = &hostFrequencyBins;
    }

    const F32* data = readableFrequencyBins->data<F32>();
    return {data, data + readableFrequencyBins->size()};
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

void RequireSpectrogramValidationError(const Registry::ModuleRegistration& impl,
                                       const Modules::Spectrogram& config,
                                       const Tensor& input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("spectrogram", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
}

std::vector<F32> ComputeSpectrogramBins(
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
    REQUIRE(Registry::BuildModule("spectrogram", implementation.device,
                                  implementation.runtime, implementation.provider,
                                  module) == Result::SUCCESS);

    Modules::Spectrogram config;
    config.height = 4;
    REQUIRE(module->create("spectrogram", config, inputs) == Result::SUCCESS);

    Runtime runtime("spectrogram", implementation.device, implementation.runtime);
    REQUIRE(runtime.create({{"spectrogram", module}}) == Result::SUCCESS);
    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

    auto bins = ReadFrequencyBins(module);
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
    return bins;
}

void RequireSpectrogramValidationError(const Registry::ModuleRegistration& impl,
                                       const Modules::Spectrogram& config,
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

    RequireSpectrogramValidationError(impl, config, input);
}

}  // namespace

TEST_CASE("Spectrogram module accepts valid height boundaries and input ranks",
          "[modules][spectrogram]") {
    auto implementations = Registry::ListAvailableModules("spectrogram");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            for (const U64 height : {U64{1}, U64{2048}}) {
                DYNAMIC_SECTION("Height: " << height) {
                    TestContext ctx("spectrogram", impl.device,
                                    impl.runtime, impl.provider);

                    Modules::Spectrogram config;
                    config.height = height;
                    config.xLabel = "Time";
                    config.yLabel = "Frequency";
                    Parser::Map serialized;
                    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
                    REQUIRE(std::any_cast<std::string>(serialized.at("xLabel")) ==
                            "Time");
                    REQUIRE(std::any_cast<std::string>(serialized.at("yLabel")) ==
                            "Frequency");
                    ctx.setConfig(config);

                    Tensor input;
                    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {64}) ==
                            Result::SUCCESS);
                    REQUIRE(SetSignalAxes(input, {.sample = Index{0}}) ==
                            Result::SUCCESS);
                    ctx.setInput("signal", input);
                    REQUIRE(ctx.run() == Result::SUCCESS);

                    Tensor batched;
                    REQUIRE(batched.create(DeviceType::CPU, DataType::F32, {2, 64}) ==
                            Result::SUCCESS);
                    REQUIRE(SetSignalAxes(batched, {
                        .sample = Index{1},
                        .batch = Index{0},
                    }) == Result::SUCCESS);
                    ctx.setInput("signal", batched);
                    REQUIRE(ctx.run() == Result::SUCCESS);

                    Tensor channels;
                    REQUIRE(channels.create(DeviceType::CPU, DataType::F32, {64}) ==
                            Result::SUCCESS);
                    REQUIRE(SetSignalAxes(channels, {.channel = Index{0}}) ==
                            Result::SUCCESS);
                    ctx.setInput("signal", channels);
                    REQUIRE(ctx.run() == Result::SUCCESS);
                }
            }
        }
    }
}

TEST_CASE("Spectrogram module rejects invalid config and inputs",
          "[modules][spectrogram][validation]") {
    auto implementations = Registry::ListAvailableModules("spectrogram");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("height must be in range") {
                Modules::Spectrogram config;
                config.height = 0;
                RequireSpectrogramValidationError(impl, config, DataType::F32, {32});

                config.height = 2049;
                RequireSpectrogramValidationError(impl, config, DataType::F32, {32});
            }

            SECTION("dtype must be F32") {
                RequireSpectrogramValidationError(impl, Modules::Spectrogram{},
                                                   DataType::CF32, {32});
            }

            SECTION("rank must be one or two") {
                RequireSpectrogramValidationError(impl, Modules::Spectrogram{},
                                                   DataType::F32, {});
                RequireSpectrogramValidationError(impl, Modules::Spectrogram{},
                                                   DataType::F32, {2, 2, 2});
            }

            SECTION("multi-axis signal roles must be present and well formed") {
                Tensor missing(impl.device, DataType::F32, {2, 32});
                RequireSpectrogramValidationError(impl, Modules::Spectrogram{}, missing);

                Tensor malformed(impl.device, DataType::F32, {32});
                REQUIRE(malformed.setAttribute(std::string(SampleAxisAttribute),
                                               I64{0}) == Result::SUCCESS);
                RequireSpectrogramValidationError(impl, Modules::Spectrogram{}, malformed);
            }

            SECTION("mixed signal roles and auxiliary dimensions are unsupported") {
                Tensor mixed(impl.device, DataType::F32, {2, 32});
                REQUIRE(SetSignalAxes(mixed, {
                    .sample = Index{1},
                    .channel = Index{0},
                }) == Result::SUCCESS);
                RequireSpectrogramValidationError(impl, Modules::Spectrogram{}, mixed);

                Tensor auxiliary(impl.device, DataType::F32, {2, 32});
                REQUIRE(SetSignalAxes(auxiliary, {.sample = Index{1}}) ==
                        Result::SUCCESS);
                RequireSpectrogramValidationError(impl, Modules::Spectrogram{}, auxiliary);
            }

            SECTION("logical render size must be supported") {
                const U64 maxRenderBinCount = std::min({
                    static_cast<U64>(std::numeric_limits<U32>::max()),
                    static_cast<U64>(std::numeric_limits<std::size_t>::max()) /
                        sizeof(F32),
                    static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) /
                        sizeof(F32),
                });
                Modules::Spectrogram config;
                config.height = 2;
                const Shape shape = {maxRenderBinCount / config.height + 1};
                RequireSpectrogramValidationError(impl, config, DataType::F32,
                                                   shape, true);
            }
        }
    }
}

TEST_CASE("Spectrogram module supports repeated computes and reconfigure",
          "[modules][spectrogram][state]") {
    auto implementations = Registry::ListAvailableModules("spectrogram");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("spectrogram", impl.device, impl.runtime, impl.provider);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {64}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {.sample = Index{0}}) == Result::SUCCESS);
            ctx.setInput("signal", input);

            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);

            Modules::Spectrogram config;
            config.height = 64;
            REQUIRE(ctx.reconfigure(config) == Result::RECREATE);
            REQUIRE(ctx.stop() == Result::SUCCESS);

            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Spectrogram indexes sample and channel batch layouts equivalently",
           "[modules][spectrogram][layout]") {
    const auto implementations = Registry::ListAvailableModules("spectrogram");
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

                    REQUIRE(ComputeSpectrogramBins(implementation, trailing) ==
                            ComputeSpectrogramBins(implementation, leading));
                }
            }
        }
    }
}
