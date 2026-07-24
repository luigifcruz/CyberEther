#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <blueprint/gain/module.hh>
#include <jetstream/registry.hh>
#include <jetstream/testing.hh>

using namespace Jetstream;

TEST_CASE("Blueprint gain module rejects unsupported input during validation",
          "[blueprint][gain][module][validation]") {
    const auto implementations =
        Registry::ListAvailableModules("blueprint_gain");
    REQUIRE_FALSE(implementations.empty());

    for (const auto& implementation : implementations) {
        for (const DataType dtype : {DataType::F64, DataType::U8}) {
            DYNAMIC_SECTION("Device: " << implementation.device
                                        << " Runtime: " << implementation.runtime
                                        << " Dtype: " << dtype) {
                Tensor input;
                REQUIRE(input.create(implementation.device, dtype, {4}) ==
                        Result::SUCCESS);

                TensorMap inputs;
                inputs["signal"].requested("test", "signal");
                inputs["signal"].tensor = input;

                std::shared_ptr<Module> module;
                REQUIRE(Registry::BuildModule("blueprint_gain",
                                              implementation.device,
                                              implementation.runtime,
                                              implementation.provider,
                                              module) == Result::SUCCESS);

                Modules::BlueprintGain config;
                REQUIRE(module->create("test", config, inputs) == Result::ERROR);
                REQUIRE(module->state() == Module::State::ERRORED);
                REQUIRE(module->outputs().empty());
            }
        }
    }
}

TEST_CASE("Blueprint gain module validates candidates without changing live config",
          "[blueprint][gain][module][reconfigure][validation]") {
    const auto implementations =
        Registry::ListAvailableModules("blueprint_gain");
    REQUIRE_FALSE(implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                                    << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {4}) ==
                    Result::SUCCESS);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("blueprint_gain",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::BlueprintGain config;
            config.gain = 2.0f;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);
            const auto outputId = module->outputs().at("signal").tensor.id();

            Parser::Map candidate;
            candidate["gain"] = F32{4.0f};
            REQUIRE(module->reconfigure(candidate, true) == Result::SUCCESS);
            REQUIRE(static_cast<const Modules::BlueprintGain&>(module->config()).gain ==
                    2.0f);
            REQUIRE(module->outputs().at("signal").tensor.id() == outputId);

            REQUIRE(module->reconfigure(candidate) == Result::SUCCESS);
            REQUIRE(static_cast<const Modules::BlueprintGain&>(module->config()).gain ==
                    4.0f);
            REQUIRE(module->outputs().at("signal").tensor.id() == outputId);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

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
