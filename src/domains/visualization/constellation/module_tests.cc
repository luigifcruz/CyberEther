#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <limits>

#include "jetstream/domains/visualization/constellation/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/testing.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

namespace {

void RequireConstellationValidationError(const Registry::ModuleRegistration& impl,
                                         const DataType dtype,
                                         const Shape& shape,
                                         const bool broadcast = false) {
    Tensor input;
    if (broadcast) {
        REQUIRE(input.create(impl.device, dtype, Shape(shape.size(), 1)) == Result::SUCCESS);
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
    REQUIRE(Registry::BuildModule("constellation", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::Constellation{}, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
}

}  // namespace

TEST_CASE("Constellation module accepts CF32 rank-1 and rank-2 inputs",
          "[modules][constellation]") {
    auto implementations = Registry::ListAvailableModules("constellation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("constellation", impl.device, impl.runtime, impl.provider);
            Modules::Constellation config;
            config.xLabel = "In Phase";
            config.yLabel = "Quadrature";
            Parser::Map serialized;
            REQUIRE(config.serialize(serialized) == Result::SUCCESS);
            REQUIRE(std::any_cast<std::string>(serialized.at("xLabel")) == "In Phase");
            REQUIRE(std::any_cast<std::string>(serialized.at("yLabel")) ==
                    "Quadrature");
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {128}) == Result::SUCCESS);

            for (U64 i = 0; i < input.size(); ++i) {
                input.at<CF32>(i) = CF32(static_cast<F32>(i), -static_cast<F32>(i));
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor batched;
            REQUIRE(batched.create(DeviceType::CPU, DataType::CF32, {4, 32}) ==
                    Result::SUCCESS);
            ctx.setInput("signal", batched);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor channels;
            REQUIRE(channels.create(DeviceType::CPU, DataType::CF32, {128}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(channels, {.channel = Index{0}}) ==
                    Result::SUCCESS);
            ctx.setInput("signal", channels);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor batchedChannels;
            REQUIRE(batchedChannels.create(DeviceType::CPU, DataType::CF32,
                                           {4, 32}) == Result::SUCCESS);
            REQUIRE(SetSignalAxes(batchedChannels, {
                .batch = Index{0},
                .channel = Index{1},
            }) == Result::SUCCESS);
            ctx.setInput("signal", batchedChannels);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor trailingBatchedChannels;
            REQUIRE(trailingBatchedChannels.create(DeviceType::CPU, DataType::CF32,
                                                   {32, 4}) == Result::SUCCESS);
            REQUIRE(SetSignalAxes(trailingBatchedChannels, {
                .batch = Index{1},
                .channel = Index{0},
            }) == Result::SUCCESS);
            ctx.setInput("signal", trailingBatchedChannels);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Constellation module rejects unsupported input dtype",
          "[modules][constellation][validation]") {
    auto implementations = Registry::ListAvailableModules("constellation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireConstellationValidationError(impl, DataType::F32, {32});
        }
    }
}

TEST_CASE("Constellation module rejects rank greater than two",
          "[modules][constellation][validation]") {
    auto implementations = Registry::ListAvailableModules("constellation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireConstellationValidationError(impl, DataType::CF32, {2, 2, 2});
        }
    }
}

TEST_CASE("Constellation module rejects rank zero during validation",
          "[modules][constellation][validation]") {
    const auto implementations = Registry::ListAvailableModules("constellation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireConstellationValidationError(impl, DataType::CF32, {});
        }
    }
}

TEST_CASE("Constellation module rejects unsupported rendering size",
          "[modules][constellation][validation][size]") {
    const auto implementations = Registry::ListAvailableModules("constellation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const U64 maxPointCount = std::min({
                static_cast<U64>(std::numeric_limits<U32>::max()),
                static_cast<U64>(std::numeric_limits<std::size_t>::max()) /
                    (32 * sizeof(F32)),
                static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) /
                    (32 * sizeof(F32)),
            });
            const Shape shape = {
                maxPointCount + 1,
            };
            RequireConstellationValidationError(impl, DataType::CF32, shape, true);
        }
    }
}

TEST_CASE("Constellation module stays stable across repeated computes",
          "[modules][constellation][state]") {
    auto implementations = Registry::ListAvailableModules("constellation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("constellation", impl.device, impl.runtime, impl.provider);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {64}) ==
                    Result::SUCCESS);

            for (U64 i = 0; i < input.size(); ++i) {
                input.at<CF32>(i) = CF32(0.1f * static_cast<F32>(i), 0.0f);
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}
