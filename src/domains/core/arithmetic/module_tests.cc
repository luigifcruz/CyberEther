#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <optional>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/arithmetic/module.hh"
#include "jetstream/memory/axis.hh"

using namespace Jetstream;

namespace {

void RequireArithmeticValidationError(const Registry::ModuleRegistration& impl,
                                      const Modules::Arithmetic& config,
                                      const DataType dtype) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, {4}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("arithmetic", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Arithmetic Module - Add F32", "[modules][arithmetic][F32]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "add";
            config.axis = 1;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            input.at(0, 0) = 1.0f; input.at(0, 1) = 2.0f; input.at(0, 2) = 3.0f;
            input.at(1, 0) = 4.0f; input.at(1, 1) = 5.0f; input.at(1, 2) = 6.0f;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 1);

            REQUIRE_THAT(out.at<F32>(0, 0),
                         Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 0),
                         Catch::Matchers::WithinAbs(15.0f, 1e-6f));
        }
    }
}

TEST_CASE("Arithmetic Module - Negative Axis", "[modules][arithmetic][F32][axis]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "add";
            REQUIRE(config.axis == -1);
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            for (U64 i = 0; i < 6; ++i) {
                input.at(i / 3, i % 3) = static_cast<F32>(i + 1);
            }
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("buffer");
            REQUIRE(out.shape() == Shape{2, 1});
            REQUIRE_THAT(out.at<F32>(0, 0),
                         Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 0),
                         Catch::Matchers::WithinAbs(15.0f, 1e-6f));
        }
    }
}

TEST_CASE("Arithmetic Module - Sub F32", "[modules][arithmetic][F32]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "sub";
            config.axis = 0;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({3});
            input.at(0) = 10.0f; input.at(1) = 3.0f; input.at(2) = 2.0f;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 1);

            // 0 - 10 - 3 - 2 = -15
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(-15.0f, 1e-6f));
        }
    }
}

TEST_CASE("Arithmetic Module - Add F32 Squeeze", "[modules][arithmetic][F32]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "add";
            config.axis = 1;
            config.squeeze = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            input.at(0, 0) = 1.0f; input.at(0, 1) = 2.0f; input.at(0, 2) = 3.0f;
            input.at(1, 0) = 4.0f; input.at(1, 1) = 5.0f; input.at(1, 2) = 6.0f;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 2);

            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(15.0f, 1e-6f));
        }
    }
}

TEST_CASE("Arithmetic Module - Add CF32", "[modules][arithmetic][CF32]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "add";
            config.axis = 0;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({3});
            input.at(0) = {1.0f, 2.0f};
            input.at(1) = {3.0f, 4.0f};
            input.at(2) = {5.0f, 6.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 1);

            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(9.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(12.0f, 1e-6f));
        }
    }
}

TEST_CASE("Arithmetic Module - Invalid Operation", "[modules][arithmetic]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Arithmetic config;
            config.operation = "invalid";
            RequireArithmeticValidationError(impl, config, DataType::F32);
        }
    }
}

TEST_CASE("Arithmetic Module - Invalid Axis", "[modules][arithmetic]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const I64 axis : {I64{5}, I64{-2}}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime
                            << " Axis: " << axis) {
                Modules::Arithmetic config;
                config.operation = "add";
                config.axis = axis;
                RequireArithmeticValidationError(impl, config, DataType::F32);
            }
        }
    }
}

TEST_CASE("Arithmetic Module - Rejects unsupported dtype during validation",
          "[modules][arithmetic][validation]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Arithmetic config;
            RequireArithmeticValidationError(impl, config, DataType::I32);
        }
    }
}

TEST_CASE("Arithmetic Module - Maps Signal Axes Across Reduction",
          "[modules][arithmetic][metadata]") {
    const auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Arithmetic config;
            config.operation = "add";
            Shape shape{2, 3, 4};
            SignalAxes inputAxes{
                .sample = Index{2},
                .batch = Index{0},
                .channel = Index{1},
            };
            SignalAxes expectedAxes = inputAxes;

            SECTION("reduction without squeeze preserves every role") {
                config.axis = 2;
            }
            SECTION("squeezing the batch axis removes only that role") {
                config.axis = 0;
                config.squeeze = true;
                expectedAxes = {
                    .sample = Index{1},
                    .channel = Index{0},
                };
            }
            SECTION("squeezing the channel axis removes only that role") {
                config.axis = 1;
                config.squeeze = true;
                expectedAxes = {
                    .sample = Index{1},
                    .batch = Index{0},
                };
            }
            SECTION("squeezing an opaque preceding axis shifts every role") {
                config.axis = 0;
                config.squeeze = true;
                shape = {2, 3, 4, 5};
                inputAxes = {
                    .sample = Index{3},
                    .batch = Index{2},
                    .channel = Index{1},
                };
                expectedAxes = {
                    .sample = Index{2},
                    .batch = Index{1},
                    .channel = Index{0},
                };
            }

            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);
            ctx.setConfig(config);
            auto input = ctx.createTensor<F32>(shape);
            REQUIRE(SetSignalAxes(input, inputAxes) == Result::SUCCESS);
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("buffer");
            SignalAxes outputAxes;
            REQUIRE(ResolveSignalAxes(output, outputAxes) == Result::SUCCESS);
            REQUIRE(outputAxes.sample == expectedAxes.sample);
            REQUIRE(outputAxes.batch == expectedAxes.batch);
            REQUIRE(outputAxes.channel == expectedAxes.channel);
        }
    }
}

TEST_CASE("Arithmetic Module - Rejects Malformed Signal Axis Metadata",
          "[modules][arithmetic][metadata][validation]") {
    const auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);
            Modules::Arithmetic config;
            config.axis = 1;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            REQUIRE(input.setAttribute(std::string(SampleAxisAttribute), Index{1}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute(std::string(ChannelAxisAttribute), I64{0}) ==
                    Result::SUCCESS);
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
