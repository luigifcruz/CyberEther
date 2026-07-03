#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/visualization/frame/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Frame module accepts valid F32 frames", "[modules][frame]") {
    auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("frame", impl.device, impl.runtime, impl.provider);

            Tensor scalar;
            REQUIRE(scalar.create(DeviceType::CPU, DataType::F32, {16, 32}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", scalar);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Modules::Frame config;
            config.lut = true;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor rgb;
            REQUIRE(rgb.create(DeviceType::CPU, DataType::F32, {16, 32, 3}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", rgb);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor rgba;
            REQUIRE(rgba.create(DeviceType::CPU, DataType::F32, {16, 32, 4}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", rgba);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Frame module rejects invalid inputs", "[modules][frame][validation]") {
    auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("dtype must be F32") {
                TestContext ctx("frame", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::U8, {16, 32}) ==
                        Result::SUCCESS);
                ctx.setInput("frame", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("rank must be two or three") {
                TestContext ctx("frame", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {32}) ==
                        Result::SUCCESS);
                ctx.setInput("frame", input);
                REQUIRE(ctx.run() == Result::ERROR);

                Tensor highRank;
                REQUIRE(highRank.create(DeviceType::CPU, DataType::F32, {2, 2, 2, 2}) ==
                        Result::SUCCESS);
                ctx.setInput("frame", highRank);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("channels must be one, three, or four") {
                TestContext ctx("frame", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {16, 32, 2}) ==
                        Result::SUCCESS);
                ctx.setInput("frame", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }
        }
    }
}

TEST_CASE("Frame module supports repeated runs and reconfigure",
          "[modules][frame][state]") {
    auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("frame", impl.device, impl.runtime, impl.provider);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {8, 8}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Modules::Frame config;
            config.lut = true;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}
