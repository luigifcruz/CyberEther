#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/fft/module.hh"

#include <cmath>

using namespace Jetstream;

TEST_CASE("FFT - DC Signal CF32", "[modules][fft][dc]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft config;
            config.forward = true;

            ctx.setConfig(config);

            // Create a DC signal (constant value).
            const U64 bufferSize = 64;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

            const F32 dcValue = 1.0f;
            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(dcValue, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // DC signal should produce a spike at bin 0.
            const F32 expectedDcBin = dcValue * static_cast<F32>(bufferSize);
            REQUIRE_THAT(std::abs(out.at<CF32>(0).real()), Catch::Matchers::WithinAbs(expectedDcBin, 1e-3f));
            REQUIRE_THAT(std::abs(out.at<CF32>(0).imag()), Catch::Matchers::WithinAbs(0.0f, 1e-3f));

            // All other bins should be near zero.
            for (U64 i = 1; i < bufferSize; ++i) {
                REQUIRE_THAT(std::abs(out.at<CF32>(i).real()), Catch::Matchers::WithinAbs(0.0f, 1e-3f));
                REQUIRE_THAT(std::abs(out.at<CF32>(i).imag()), Catch::Matchers::WithinAbs(0.0f, 1e-3f));
            }
        }
    }
}

TEST_CASE("FFT - Forward/Inverse Roundtrip CF32", "[modules][fft][roundtrip]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            // Forward FFT.
            TestContext forwardCtx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft forwardConfig;
            forwardConfig.forward = true;

            forwardCtx.setConfig(forwardConfig);

            // Create a test signal.
            const U64 bufferSize = 64;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                const F64 t = static_cast<F64>(i) / static_cast<F64>(bufferSize);
                input.at<CF32>(i) = CF32(static_cast<F32>(std::cos(2.0 * JST_PI * 4.0 * t)),
                                         static_cast<F32>(std::sin(2.0 * JST_PI * 4.0 * t)));
            }

            forwardCtx.setInput("signal", input);
            REQUIRE(forwardCtx.run() == Result::SUCCESS);

            // Inverse FFT.
            TestContext inverseCtx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft inverseConfig;
            inverseConfig.forward = false;

            inverseCtx.setConfig(inverseConfig);
            inverseCtx.setInput("signal", forwardCtx.output("signal"));

            REQUIRE(inverseCtx.run() == Result::SUCCESS);

            auto& recovered = inverseCtx.output("signal");

            // After forward+inverse, signal should be recovered (scaled by N).
            for (U64 i = 0; i < bufferSize; ++i) {
                const F32 scale = static_cast<F32>(bufferSize);
                const F32 expectedReal = input.at<CF32>(i).real() * scale;
                const F32 expectedImag = input.at<CF32>(i).imag() * scale;
                REQUIRE_THAT(recovered.at<CF32>(i).real(), Catch::Matchers::WithinAbs(expectedReal, 1e-2f));
                REQUIRE_THAT(recovered.at<CF32>(i).imag(), Catch::Matchers::WithinAbs(expectedImag, 1e-2f));
            }
        }
    }
}
