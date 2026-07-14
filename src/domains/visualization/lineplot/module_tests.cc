#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/visualization/lineplot/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

#include "module_impl.hh"

using namespace Jetstream;

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
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor batched;
            REQUIRE(batched.create(DeviceType::CPU, DataType::F32, {2, 128}) ==
                    Result::SUCCESS);
            ctx.setInput("signal", batched);
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
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {64}) ==
                    Result::SUCCESS);

            SECTION("averaging must be positive") {
                TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);
                Modules::Lineplot config;
                config.averaging = 0;
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("decimation must be positive") {
                TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);
                Modules::Lineplot config;
                config.decimation = 0;
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("grid dimensions must be at least two") {
                TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);
                Modules::Lineplot config;
                config.numberOfVerticalLines = 1;
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);

                config.numberOfVerticalLines = 11;
                config.numberOfHorizontalLines = 1;
                ctx.setConfig(config);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("thickness must be positive") {
                TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);
                Modules::Lineplot config;
                config.thickness = 0.0f;
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
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
                TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {64}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("rank must be one or two") {
                TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2, 2, 2}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("effective number of elements must be at least two") {
                TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {1}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }
        }
    }
}

TEST_CASE("Lineplot module handles repeated runs and config updates",
          "[modules][lineplot][state]") {
    auto implementations = Registry::ListAvailableModules("lineplot");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("lineplot", impl.device, impl.runtime, impl.provider);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {64}) ==
                    Result::SUCCESS);
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Modules::Lineplot config;
            config.averaging = 8;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);

            config.decimation = 2;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);
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
                batch, index, shape[1], decimation)];
        }
    }

    REQUIRE(sums[0] == 11.0f);
    REQUIRE(sums[1] == 33.0f);
}
