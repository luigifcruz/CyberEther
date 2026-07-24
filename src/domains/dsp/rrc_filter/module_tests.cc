#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/rrc_filter/module.hh"

#include <cmath>
#include <limits>

using namespace Jetstream;

namespace {

void RequireRrcFilterValidationError(const Registry::ModuleRegistration& impl,
                                     const Modules::RrcFilter& config,
                                     const DataType dtype) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, {64}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("rrc_filter", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("RRC Filter - CF32 Impulse Response",
          "[modules][rrc_filter][cf32]") {
    auto implementations =
        Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("rrc_filter", impl.device,
                           impl.runtime, impl.provider);

            Modules::RrcFilter config;
            config.symbolRate = 1.0e6f;
            config.sampleRate = 4.0e6f;
            config.rollOff = 0.35f;
            config.taps = 11;

            ctx.setConfig(config);

            // Create impulse input: 1 at index 0, zeros elsewhere.
            const U64 bufferSize = 32;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(0.0f, 0.0f);
            }
            input.at<CF32>(0) = CF32(1.0f, 0.0f);

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // The impulse response should be the filter
            // coefficients (delayed by taps/2). The peak should be
            // near the center of the filter at index taps/2 = 5.
            // Verify that the output is not all zeros.
            F32 maxMag = 0.0f;
            U64 maxIdx = 0;
            for (U64 i = 0; i < bufferSize; ++i) {
                F32 mag = std::abs(out.at<CF32>(i));
                if (mag > maxMag) {
                    maxMag = mag;
                    maxIdx = i;
                }
            }

            // Peak should be at index taps/2 = 5.
            REQUIRE(maxIdx == config.taps / 2);
            REQUIRE(maxMag > 0.0f);
        }
    }
}

TEST_CASE("RRC Filter - F32 DC Passthrough",
          "[modules][rrc_filter][f32]") {
    auto implementations =
        Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("rrc_filter", impl.device,
                           impl.runtime, impl.provider);

            Modules::RrcFilter config;
            config.symbolRate = 1.0e6f;
            config.sampleRate = 4.0e6f;
            config.rollOff = 0.35f;
            config.taps = 11;

            ctx.setConfig(config);

            // Create constant DC input.
            const U64 bufferSize = 64;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<F32>(i) = 1.0f;
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // After the filter settles (past taps-1 samples),
            // the output should converge to the sum of all
            // coefficients times the DC value.
            // Verify the tail samples are approximately equal.
            const F32 tailValue = out.at<F32>(bufferSize - 1);
            for (U64 i = bufferSize - 10; i < bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i),
                    Catch::Matchers::WithinRel(tailValue, 0.01f));
            }
        }
    }
}

TEST_CASE("RRC Filter - Invalid Even Taps",
          "[modules][rrc_filter][error]") {
    auto implementations =
        Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("rrc_filter", impl.device,
                           impl.runtime, impl.provider);

            Modules::RrcFilter config;
            config.taps = 10;  // Even number, should fail.

            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {64}) == Result::SUCCESS);

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() != Result::SUCCESS);
        }
    }
}

TEST_CASE("RRC Filter - Invalid Sample Rate",
          "[modules][rrc_filter][error]") {
    auto implementations =
        Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("rrc_filter", impl.device,
                           impl.runtime, impl.provider);

            Modules::RrcFilter config;
            config.symbolRate = 2.0e6f;
            config.sampleRate = 1.0e6f;  // Less than symbol rate.

            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {64}) == Result::SUCCESS);

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() != Result::SUCCESS);
        }
    }
}

TEST_CASE("RRC Filter - Rejects invalid candidates during validation",
          "[modules][rrc_filter][validation]") {
    auto implementations = Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("numeric configuration must be finite") {
            Modules::RrcFilter config;
            config.symbolRate = std::numeric_limits<F32>::quiet_NaN();
            RequireRrcFilterValidationError(impl, config, DataType::F32);
        }

        SECTION("symbol rate must be positive") {
            Modules::RrcFilter config;
            config.symbolRate = 0.0f;
            RequireRrcFilterValidationError(impl, config, DataType::F32);
        }

        SECTION("sample rate must be positive") {
            Modules::RrcFilter config;
            config.sampleRate = 0.0f;
            RequireRrcFilterValidationError(impl, config, DataType::F32);
        }

        SECTION("roll-off must be finite") {
            Modules::RrcFilter config;
            config.rollOff = std::numeric_limits<F32>::quiet_NaN();
            RequireRrcFilterValidationError(impl, config, DataType::F32);
        }

        SECTION("roll-off cannot be negative") {
            Modules::RrcFilter config;
            config.rollOff = -0.01f;
            RequireRrcFilterValidationError(impl, config, DataType::F32);
        }

        SECTION("roll-off cannot exceed one") {
            Modules::RrcFilter config;
            config.rollOff = 1.01f;
            RequireRrcFilterValidationError(impl, config, DataType::F32);
        }

        SECTION("tap count must be at least three") {
            Modules::RrcFilter config;
            config.taps = 1;
            RequireRrcFilterValidationError(impl, config, DataType::F32);
        }

        SECTION("input dtype must be supported") {
            Modules::RrcFilter config;
            RequireRrcFilterValidationError(impl, config, DataType::U8);
        }

        SECTION("tap buffer size must be representable") {
            Modules::RrcFilter config;
            config.taps = std::numeric_limits<U64>::max();
            RequireRrcFilterValidationError(impl, config, DataType::F32);
        }
    }
}
