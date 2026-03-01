#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/visualization/spectrogram/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Spectrogram module accepts valid inputs", "[modules][spectrogram]") {
    auto implementations = Registry::ListAvailableModules("spectrogram");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("spectrogram", impl.device, impl.runtime, impl.provider);

            Modules::Spectrogram config;
            config.height = 128;
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

TEST_CASE("Spectrogram module rejects invalid config and inputs",
          "[modules][spectrogram][validation]") {
    auto implementations = Registry::ListAvailableModules("spectrogram");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("height must be in range") {
                TestContext ctx("spectrogram", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {32}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", input);

                Modules::Spectrogram config;
                config.height = 0;
                ctx.setConfig(config);
                REQUIRE(ctx.run() == Result::ERROR);

                config.height = 2049;
                ctx.setConfig(config);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("dtype must be F32") {
                TestContext ctx("spectrogram", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {32}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("rank must be one or two") {
                TestContext ctx("spectrogram", impl.device, impl.runtime, impl.provider);
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2, 2, 2}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }
        }
    }
}

TEST_CASE("Spectrogram module supports repeated runs and reconfigure",
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

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Modules::Spectrogram config;
            config.height = 64;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}
