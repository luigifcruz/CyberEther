#include <catch2/catch_test_macros.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/adsb/module.hh"

using namespace Jetstream;

TEST_CASE("ADS-B - Silence Input",
          "[modules][adsb][silence]") {
    auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("adsb", impl.device, impl.runtime,
                           impl.provider);

            Modules::Adsb config;
            ctx.setConfig(config);

            const U64 bufferSize = 8192;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

        }
    }
}

TEST_CASE("ADS-B - Random Noise Input",
          "[modules][adsb][noise]") {
    auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("adsb", impl.device, impl.runtime,
                           impl.provider);

            Modules::Adsb config;
            ctx.setConfig(config);

            // Create low-level noise input.
            const U64 bufferSize = 65536;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                const F32 r = static_cast<F32>(i % 7) * 0.001f - 0.003f;
                const F32 q = static_cast<F32>(i % 11) * 0.001f - 0.005f;
                input.at<CF32>(i) = CF32(r, q);
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

        }
    }
}

TEST_CASE("ADS-B - Invalid Input DType",
          "[modules][adsb][error]") {
    auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("adsb", impl.device, impl.runtime, impl.provider);

            Modules::Adsb config;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32,
                                 {8192}) == Result::SUCCESS);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
