#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>

#include "jetstream/domains/dsp/phase_correction/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Phase correction advances across batches and submissions",
          "[modules][phase_correction][batch]") {
    const auto implementations = Registry::ListAvailableModules("phase_correction");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("phase_correction", impl.device, impl.runtime, impl.provider);
            Modules::PhaseCorrection config;
            config.phaseIncrement = JST_PI / 2.0;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2, 3});
            REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            for (U64 batch = 0; batch < input.shape(0); ++batch) {
                for (U64 sample = 0; sample < input.shape(1); ++sample) {
                    input.at(batch, sample) = CF32{1.0f, 0.0f};
                }
            }
            ctx.setInput("signal", input);

            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            for (U64 sample = 0; sample < output.shape(1); ++sample) {
                REQUIRE_THAT(output.at<CF32>(0, sample).real(),
                             Catch::Matchers::WithinAbs(1.0f, 1e-5f));
                REQUIRE_THAT(output.at<CF32>(1, sample).imag(),
                             Catch::Matchers::WithinAbs(1.0f, 1e-5f));
            }

            REQUIRE(ctx.compute() == Result::SUCCESS);
            for (U64 sample = 0; sample < output.shape(1); ++sample) {
                REQUIRE_THAT(output.at<CF32>(0, sample).real(),
                             Catch::Matchers::WithinAbs(-1.0f, 1e-5f));
                REQUIRE_THAT(output.at<CF32>(1, sample).imag(),
                             Catch::Matchers::WithinAbs(-1.0f, 1e-5f));
            }
        }
    }
}

TEST_CASE("Phase correction handles large finite increments",
          "[modules][phase_correction][batch][numeric]") {
    const auto implementations = Registry::ListAvailableModules("phase_correction");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("phase_correction", impl.device, impl.runtime, impl.provider);
            Modules::PhaseCorrection config;
            config.phaseIncrement = std::numeric_limits<F64>::max();
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2, 1});
            REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            input.at(0, 0) = CF32{1.0f, 0.0f};
            input.at(1, 0) = CF32{1.0f, 0.0f};
            ctx.setInput("signal", input);

            REQUIRE(ctx.start() == Result::SUCCESS);
            const F64 increment =
                std::remainder(config.phaseIncrement, 2.0 * JST_PI);
            for (U64 submission = 0; submission < 2; ++submission) {
                REQUIRE(ctx.compute() == Result::SUCCESS);
                const auto& output = ctx.output("signal");
                for (U64 batch = 0; batch < input.shape(0); ++batch) {
                    const F64 phase = std::remainder(
                        increment * static_cast<F64>(submission * input.shape(0) + batch),
                        2.0 * JST_PI);
                    REQUIRE_THAT(output.at<CF32>(batch, 0).real(),
                                 Catch::Matchers::WithinAbs(
                                     static_cast<F32>(std::cos(phase)), 1e-5f));
                    REQUIRE_THAT(output.at<CF32>(batch, 0).imag(),
                                 Catch::Matchers::WithinAbs(
                                     static_cast<F32>(std::sin(phase)), 1e-5f));
                }
            }
        }
    }
}

TEST_CASE("Phase correction rejects invalid configuration and dtype",
          "[modules][phase_correction][validation]") {
    const auto implementations = Registry::ListAvailableModules("phase_correction");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("phase_correction", impl.device, impl.runtime, impl.provider);
            Modules::PhaseCorrection config;

            SECTION("non-finite increment") {
                config.phaseIncrement = std::numeric_limits<F64>::infinity();
                auto input = ctx.createTensor<CF32>({4});
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("non-complex input") {
                auto input = ctx.createTensor<F32>({4});
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }
        }
    }
}
