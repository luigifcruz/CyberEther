#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/filter_taps/module.hh"

#include <cmath>
#include <complex>
#include <limits>
#include <unordered_set>

using namespace Jetstream;

namespace {

void RequireFilterTapsValidationError(const Registry::ModuleRegistration& impl,
                                      const Modules::FilterTaps& config) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("filter_taps", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Filter Taps - Default Config", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 1.0e6;
            config.center = {0.0};
            config.taps = 101;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");
            REQUIRE(out.size() == 101);
            REQUIRE(out.dtype() == DataType::CF32);

            // Center tap should be the maximum magnitude.
            const U64 centerIdx = 50;
            F32 centerMag = std::abs(out.at<CF32>(0, centerIdx));
            for (U64 i = 0; i < 101; ++i) {
                REQUIRE(std::abs(out.at<CF32>(0, i)) <= centerMag + 1e-6f);
            }
        }
    }
}

TEST_CASE("Filter Taps - Zero Center Symmetry", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 0.5e6;
            config.center = {0.0};
            config.taps = 51;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");

            // With zero center, coefficients should be real-valued (imaginary ~0).
            for (U64 i = 0; i < config.taps; ++i) {
                REQUIRE_THAT(out.at<CF32>(0, i).imag(),
                             Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            }

            // Coefficients should be symmetric around center.
            const U64 center = config.taps / 2;
            for (U64 i = 0; i < center; ++i) {
                REQUIRE_THAT(out.at<CF32>(0, i).real(),
                             Catch::Matchers::WithinAbs(
                                 out.at<CF32>(0, config.taps - 1 - i).real(),
                                 1e-6f));
            }
        }
    }
}

TEST_CASE("Filter Taps - Tensor Attributes", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 0.5e6;
            config.center = {0.1e6};
            config.taps = 51;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");

            REQUIRE(out.hasAttribute("sampleRate"));
            REQUIRE(out.hasAttribute("bandwidth"));
            REQUIRE(out.hasAttribute("center"));
            REQUIRE(out.hasAttribute("sampleAxis"));
            REQUIRE(out.attribute("sampleAxis").type() == typeid(Index));
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{1});
            REQUIRE(out.hasAttribute("channelAxis"));
            REQUIRE(out.attribute("channelAxis").type() == typeid(Index));
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{0});
            REQUIRE_FALSE(out.hasAttribute("batchAxis"));

            REQUIRE_THAT(std::any_cast<F32>(out.attribute("sampleRate")),
                         Catch::Matchers::WithinAbs(2.0e6f, 1.0f));
            REQUIRE_THAT(std::any_cast<F32>(out.attribute("bandwidth")),
                         Catch::Matchers::WithinAbs(0.5e6f, 1.0f));
            REQUIRE_THAT(std::any_cast<F32>(out.attribute("center")),
                         Catch::Matchers::WithinAbs(0.1e6f, 1.0f));
        }
    }
}

TEST_CASE("Filter Taps - Multi-Head", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 0.2e6;
            config.center = {0.0, 0.2e6, -0.4e6};
            config.taps = 51;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");

            // Output should be 2D: {3, 51}.
            REQUIRE(out.rank() == 2);
            REQUIRE(out.shape(0) == 3);
            REQUIRE(out.shape(1) == 51);
            REQUIRE(std::any_cast<std::vector<F32>>(out.attribute("center")) ==
                    std::vector<F32>{0.0f, 0.2e6f, -0.4e6f});

            // First head (center=0) should have real-valued coefficients.
            for (U64 i = 0; i < config.taps; ++i) {
                REQUIRE_THAT(out.at<CF32>(0, i).imag(),
                             Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            }

            // Second head (center=0.2MHz) should have complex coefficients.
            bool hasComplex = false;
            for (U64 i = 0; i < config.taps; ++i) {
                if (std::abs(out.at<CF32>(1, i).imag()) > 1e-6f) {
                    hasComplex = true;
                    break;
                }
            }
            REQUIRE(hasComplex);
        }
    }
}

TEST_CASE("Filter Taps - Center Tap Matches Normalized Bandwidth", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 0.2e6;
            config.center = {0.0};
            config.taps = 101;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");
            const U64 centerIdx = config.taps / 2;

            const F32 expectedCenter = static_cast<F32>(config.bandwidth / config.sampleRate);
            REQUIRE_THAT(out.at<CF32>(0, centerIdx).real(),
                         Catch::Matchers::WithinRel(expectedCenter, 1e-4f));
            REQUIRE_THAT(out.at<CF32>(0, centerIdx).imag(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Filter Taps - Valid Coefficients Include Center Phase",
          "[modules][filter_taps][coefficients]") {
    const auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 8.0;
            config.bandwidth = 2.0;
            config.center = {1.0};
            config.taps = 5;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& out = ctx.output("coeffs");
            const F32 side = static_cast<F32>(0.17 / JST_PI);
            const CF32 expected[] = {
                {0.0f, 0.0f}, {side, -side}, {0.25f, 0.0f},
                {side, side}, {0.0f, 0.0f},
            };
            for (U64 tap = 0; tap < config.taps; ++tap) {
                REQUIRE_THAT(out.at<CF32>(0, tap).real(),
                             Catch::Matchers::WithinAbs(expected[tap].real(), 1e-6f));
                REQUIRE_THAT(out.at<CF32>(0, tap).imag(),
                             Catch::Matchers::WithinAbs(expected[tap].imag(), 1e-6f));
            }
        }
    }
}

TEST_CASE("Filter Taps - One Tap Uses A Unity Singleton Window",
          "[modules][filter_taps][coefficients][singleton]") {
    const auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 8.0;
            config.bandwidth = 2.0;
            config.center = {0.0, 3.0, -3.0};
            config.taps = 1;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& out = ctx.output("coeffs");
            REQUIRE(out.shape() == Shape{3, 1});
            for (U64 head = 0; head < config.center.size(); ++head) {
                REQUIRE_THAT(out.at<CF32>(head, 0).real(),
                             Catch::Matchers::WithinAbs(0.25f, 1e-6f));
                REQUIRE_THAT(out.at<CF32>(head, 0).imag(),
                             Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            }
        }
    }
}

TEST_CASE("Filter Taps - Metadata Must Be Representable As F32 During Validation",
          "[modules][filter_taps][validation][metadata]") {
    const auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("sample rate cannot narrow to zero or overflow F32") {
                Modules::FilterTaps config;
                config.sampleRate = std::numeric_limits<F64>::denorm_min();
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.sampleRate = std::numeric_limits<F64>::max();
                RequireFilterTapsValidationError(impl, config);
            }

            SECTION("bandwidth cannot narrow to zero or overflow F32") {
                Modules::FilterTaps config;
                config.sampleRate = 1.0;
                config.bandwidth = std::numeric_limits<F64>::denorm_min();
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.bandwidth = std::numeric_limits<F64>::max();
                RequireFilterTapsValidationError(impl, config);
            }

            SECTION("nonzero centers cannot narrow to zero or overflow F32") {
                Modules::FilterTaps config;
                config.sampleRate = 1.0;
                config.bandwidth = 1.0;
                config.center = {std::numeric_limits<F64>::denorm_min()};
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.center = {std::numeric_limits<F64>::max()};
                RequireFilterTapsValidationError(impl, config);
            }
        }
    }
}

TEST_CASE("Filter Taps - Invalid Candidates Stop During Validation",
          "[modules][filter_taps][validation]") {
    const auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("sample rate must be finite and positive") {
                Modules::FilterTaps config;
                config.sampleRate = std::numeric_limits<F64>::quiet_NaN();
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.sampleRate = 0.0;
                RequireFilterTapsValidationError(impl, config);
            }

            SECTION("bandwidth must be finite, positive, and no greater than sample rate") {
                Modules::FilterTaps config;
                config.bandwidth = std::numeric_limits<F64>::infinity();
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.bandwidth = 0.0;
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.bandwidth = config.sampleRate + 1.0;
                RequireFilterTapsValidationError(impl, config);
            }

            SECTION("centers must be finite, present, and within Nyquist") {
                Modules::FilterTaps config;
                config.center = {std::numeric_limits<F64>::quiet_NaN()};
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.center = {};
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.center = {config.sampleRate / 2.0 + 1.0};
                RequireFilterTapsValidationError(impl, config);
            }

            SECTION("tap count must be nonzero and odd") {
                Modules::FilterTaps config;
                config.taps = 0;
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.taps = 2;
                RequireFilterTapsValidationError(impl, config);
            }

            SECTION("output geometry must be representable") {
                Modules::FilterTaps config;
                config.center = {0.0, 1.0};
                config.taps = std::numeric_limits<U64>::max();
                RequireFilterTapsValidationError(impl, config);

                config = {};
                config.taps = std::numeric_limits<U64>::max();
                RequireFilterTapsValidationError(impl, config);
            }

            SECTION("output allocation must be representable") {
                Modules::FilterTaps config;
                config.taps = std::numeric_limits<U64>::max() /
                              static_cast<U64>(sizeof(CF32));
                RequireFilterTapsValidationError(impl, config);
            }
        }
    }
}

TEST_CASE("Filter Taps - direct runtime rematerializes output",
          "[modules][filter_taps][state]") {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("filter_taps",
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  "generic",
                                  module) == Result::SUCCESS);

    Modules::FilterTaps config;
    config.sampleRate = 2.0e6;
    config.bandwidth = 0.5e6;
    config.center = {0.0};
    config.taps = 51;
    REQUIRE(module->create("filter_taps_rematerialize", config, {}) == Result::SUCCESS);

    Runtime runtime("filter_taps_rematerialize", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"filter_taps_rematerialize", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"filter_taps_rematerialize"}, skippedModules, failedModules) ==
            Result::SUCCESS);

    Tensor output = module->outputs().at("coeffs").tensor;
    const CF32 expected = output.at<CF32>(0, 17);
    output.at<CF32>(0, 17) = CF32(123.0f, 456.0f);

    REQUIRE(runtime.compute({"filter_taps_rematerialize"}, skippedModules, failedModules) ==
            Result::SUCCESS);
    REQUIRE(output.at<CF32>(0, 17) == expected);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}
