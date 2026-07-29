#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <limits>

#include "jetstream/domains/visualization/spectrogram/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

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
