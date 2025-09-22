#include <catch2/catch_test_macros.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("Constellation module accepts CF32 rank-1 and rank-2 inputs",
          "[modules][constellation]") {
    auto implementations = Registry::ListAvailableModules("constellation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("constellation", impl.device, impl.runtime, impl.provider);

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
        }
    }
}

TEST_CASE("Constellation module rejects unsupported input dtype",
          "[modules][constellation][validation]") {
    auto implementations = Registry::ListAvailableModules("constellation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("constellation", impl.device, impl.runtime, impl.provider);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {32}) == Result::SUCCESS);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Constellation module rejects rank greater than two",
          "[modules][constellation][validation]") {
    auto implementations = Registry::ListAvailableModules("constellation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("constellation", impl.device, impl.runtime, impl.provider);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {2, 2, 2}) ==
                    Result::SUCCESS);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Constellation module stays stable across repeated runs",
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
            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}
