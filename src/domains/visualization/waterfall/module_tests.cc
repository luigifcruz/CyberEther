#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/visualization/waterfall/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

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
