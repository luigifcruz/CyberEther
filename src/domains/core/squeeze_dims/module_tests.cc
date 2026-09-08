#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <any>
#include <optional>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/squeeze_dims/module.hh"
#include "jetstream/memory/axis.hh"

using namespace Jetstream;

namespace {

void RequireSqueezeDimsValidationError(const Registry::ModuleRegistration& impl,
                                       const I64 axis,
                                       const Shape& shape,
                                       const std::string& attribute = {},
                                       const std::any& attributeValue = {}) {
    Tensor input;
    REQUIRE(input.create(impl.device, DataType::F32, shape) == Result::SUCCESS);
    if (attributeValue.has_value()) {
        if (attribute != SampleAxisAttribute) {
            REQUIRE(input.setAttribute(std::string(SampleAxisAttribute), Index{0}) ==
                    Result::SUCCESS);
        }
        REQUIRE(input.setAttribute(attribute, attributeValue) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("squeeze_dims", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);

    Modules::SqueezeDims config;
    config.axis = axis;
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

void RequireSqueezedSignalAxes(const Registry::ModuleRegistration& impl,
                               const I64 axis,
                               const Shape& shape,
                               const SignalAxes& inputAxes,
                               const SignalAxes& expectedAxes) {
    TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

    Modules::SqueezeDims config;
    config.axis = axis;
    ctx.setConfig(config);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, shape) == Result::SUCCESS);
    REQUIRE(SetSignalAxes(input, inputAxes) == Result::SUCCESS);
    ctx.setInput("buffer", input);

    REQUIRE(ctx.run() == Result::SUCCESS);

    const auto& out = ctx.output("buffer");
    SignalAxes outputAxes;
    REQUIRE(ResolveSignalAxes(out, outputAxes) == Result::SUCCESS);
    REQUIRE(outputAxes.sample == expectedAxes.sample);
    REQUIRE(outputAxes.batch == expectedAxes.batch);
    REQUIRE(outputAxes.channel == expectedAxes.channel);
}

}  // namespace

TEST_CASE("SqueezeDims Module - Squeeze 2D to 1D at axis 0 F32", "[modules][squeeze_dims][F32]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({1, 4});
            for (U64 i = 0; i < 4; ++i) {
                input.at(0, i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("SqueezeDims Module - Squeeze 2D to 1D at axis 1 F32", "[modules][squeeze_dims][F32]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 1;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 1});
            for (U64 i = 0; i < 4; ++i) {
                input.at(i, 0) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("SqueezeDims Module - Negative Axis", "[modules][squeeze_dims][F32][axis]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            REQUIRE(config.axis == -1);
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 1});
            for (U64 i = 0; i < 4; ++i) {
                input.at(i, 0) = static_cast<F32>(i);
            }
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("buffer");
            REQUIRE(out.shape() == Shape{4});
            for (U64 i = 0; i < 4; ++i) {
                REQUIRE(out.at<F32>(i) == static_cast<F32>(i));
            }
        }
    }
}

TEST_CASE("SqueezeDims Module - Squeeze 3D to 2D at axis 1 F32", "[modules][squeeze_dims][F32]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 1;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 1, 4});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i / 4, 0, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 4);

            for (U64 i = 0; i < 8; ++i) {
                REQUIRE_THAT(out.at<F32>(i / 4, i % 4),
                             Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("SqueezeDims Module - CF32", "[modules][squeeze_dims][CF32]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({1, 4});
            input.at(0, 0) = {0.0f, 1.0f};
            input.at(0, 1) = {2.0f, 3.0f};
            input.at(0, 2) = {4.0f, 5.0f};
            input.at(0, 3) = {6.0f, 7.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            REQUIRE_THAT(out.at<CF32>(0).real(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(2).real(), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(2).imag(), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(3).real(), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(3).imag(), Catch::Matchers::WithinAbs(7.0f, 1e-6f));
        }
    }
}

TEST_CASE("SqueezeDims Module - Axis Out of Range Error", "[modules][squeeze_dims][error]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const I64 axis : {I64{5}, I64{-3}}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime
                            << " Axis: " << axis) {
                RequireSqueezeDimsValidationError(impl, axis, {1, 4});
            }
        }
    }
}

TEST_CASE("SqueezeDims Module - Dimension Not Size 1 Error", "[modules][squeeze_dims][error]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireSqueezeDimsValidationError(impl, 1, {1, 4});
        }
    }
}

TEST_CASE("SqueezeDims Module - Remaps And Removes Signal Axes",
          "[modules][squeeze_dims][metadata]") {
    const auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireSqueezedSignalAxes(
                impl, 0, {1, 2, 3, 4},
                {.sample = Index{3}, .batch = Index{1}, .channel = Index{2}},
                {.sample = Index{2}, .batch = Index{0}, .channel = Index{1}});
            RequireSqueezedSignalAxes(
                impl, 3, {2, 3, 4, 1},
                {.sample = Index{1}, .batch = Index{0}, .channel = Index{2}},
                {.sample = Index{1}, .batch = Index{0}, .channel = Index{2}});
            RequireSqueezedSignalAxes(
                impl, 1, {2, 1, 3},
                {.sample = Index{2}, .batch = Index{0}, .channel = Index{1}},
                {.sample = Index{1}, .batch = Index{0}});
        }
    }
}

TEST_CASE("SqueezeDims Module - Invalid Signal Axis Metadata Error",
          "[modules][squeeze_dims][metadata][error]") {
    const auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireSqueezeDimsValidationError(
                impl, 0, {1, 4}, std::string(SampleAxisAttribute), I64{0});
            RequireSqueezeDimsValidationError(
                impl, 0, {1, 4}, std::string(ChannelAxisAttribute), Index{2});
        }
    }
}
