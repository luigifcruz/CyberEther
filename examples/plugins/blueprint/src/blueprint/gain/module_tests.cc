#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <blueprint/gain/module.hh>
#include <jetstream/registry.hh>
#include <jetstream/testing.hh>

using namespace Jetstream;

TEST_CASE("Blueprint gain module scales F32 samples",
          "[blueprint][gain][module][f32]") {
    const auto implementations =
        Registry::ListAvailableModules("blueprint_gain");
    REQUIRE_FALSE(implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            TestContext context("blueprint_gain",
                                implementation.device,
                                implementation.runtime,
                                implementation.provider);

            Modules::BlueprintGain config;
            config.gain = 2.5f;
            context.setConfig(config);

            auto input = context.createTensor<F32>({4});
            input.at(0) = 1.0f;
            input.at(1) = -2.0f;
            input.at(2) = 0.5f;
            input.at(3) = 4.0f;
            context.setInput("signal", input);

            REQUIRE(context.run() == Result::SUCCESS);

            const auto& output = context.output("signal");
            REQUIRE(output.shape() == Shape{4});
            REQUIRE(output.dtype() == DataType::F32);
            REQUIRE_THAT(output.at<F32>(0),
                         Catch::Matchers::WithinAbs(2.5f, 1e-6f));
            REQUIRE_THAT(output.at<F32>(1),
                         Catch::Matchers::WithinAbs(-5.0f, 1e-6f));
            REQUIRE_THAT(output.at<F32>(2),
                         Catch::Matchers::WithinAbs(1.25f, 1e-6f));
            REQUIRE_THAT(output.at<F32>(3),
                         Catch::Matchers::WithinAbs(10.0f, 1e-6f));
        }
    }
}

TEST_CASE("Blueprint gain module scales CF32 samples",
          "[blueprint][gain][module][cf32]") {
    const auto implementations =
        Registry::ListAvailableModules("blueprint_gain");
    REQUIRE_FALSE(implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            TestContext context("blueprint_gain",
                                implementation.device,
                                implementation.runtime,
                                implementation.provider);

            Modules::BlueprintGain config;
            config.gain = 0.5f;
            context.setConfig(config);

            auto input = context.createTensor<CF32>({2});
            input.at(0) = {2.0f, 4.0f};
            input.at(1) = {-6.0f, 8.0f};
            context.setInput("signal", input);

            REQUIRE(context.run() == Result::SUCCESS);

            const auto& output = context.output("signal");
            REQUIRE(output.shape() == Shape{2});
            REQUIRE(output.dtype() == DataType::CF32);
            REQUIRE_THAT(output.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(output.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(output.at<CF32>(1).real(),
                         Catch::Matchers::WithinAbs(-3.0f, 1e-6f));
            REQUIRE_THAT(output.at<CF32>(1).imag(),
                         Catch::Matchers::WithinAbs(4.0f, 1e-6f));
        }
    }
}
