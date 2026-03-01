#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/visualization/lineplot/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

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
