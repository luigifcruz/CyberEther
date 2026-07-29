#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/domains/core/reshape/module.hh"

using namespace Jetstream;

namespace {

void RequireReshapeValidationError(const Registry::ModuleRegistration& impl,
                                   const std::string& targetShape,
                                   const Shape& inputShape) {
    Tensor input;
    REQUIRE(input.create(impl.device, DataType::F32, inputShape) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    Modules::Reshape config;
    config.shape = targetShape;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("reshape", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);

    Result result = Result::SUCCESS;
    REQUIRE_NOTHROW(result = module->create("test", config, inputs));
    REQUIRE(result == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Reshape Module - Flatten 2D to 1D F32", "[modules][reshape][F32]") {
    auto implementations = Registry::ListAvailableModules("reshape");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("reshape", impl.device, impl.runtime, impl.provider);

            Modules::Reshape config;
            config.shape = "[8]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 4});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 8);

            for (U64 i = 0; i < 8; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("Reshape Module - Unflatten 1D to 2D F32", "[modules][reshape][F32]") {
    auto implementations = Registry::ListAvailableModules("reshape");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("reshape", impl.device, impl.runtime, impl.provider);

            Modules::Reshape config;
            config.shape = "[4, 4]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({16});
            for (U64 i = 0; i < 16; ++i) {
                input.at(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 4);
            REQUIRE(out.shape(1) == 4);

            for (U64 i = 0; i < 16; ++i) {
                REQUIRE_THAT(out.at<F32>(i / 4, i % 4),
                             Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("Reshape Module - Reshape 2D F32", "[modules][reshape][F32][2d]") {
    auto implementations = Registry::ListAvailableModules("reshape");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("reshape", impl.device, impl.runtime, impl.provider);

            Modules::Reshape config;
            config.shape = "[4, 2]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 4});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 4);
            REQUIRE(out.shape(1) == 2);

            // Data order should be preserved (row-major).
            for (U64 i = 0; i < 8; ++i) {
                REQUIRE_THAT(out.at<F32>(i / 2, i % 2),
                             Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("Reshape Module - Reshape to 3D F32", "[modules][reshape][F32][3d]") {
    auto implementations = Registry::ListAvailableModules("reshape");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("reshape", impl.device, impl.runtime, impl.provider);

            Modules::Reshape config;
            config.shape = "[2, 2, 4]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({16});
            for (U64 i = 0; i < 16; ++i) {
                input.at(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 3);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 2);
            REQUIRE(out.shape(2) == 4);

            U64 idx = 0;
            for (U64 i = 0; i < 2; ++i) {
                for (U64 j = 0; j < 2; ++j) {
                    for (U64 k = 0; k < 4; ++k) {
                        REQUIRE_THAT(out.at<F32>(i, j, k),
                                     Catch::Matchers::WithinAbs(static_cast<F32>(idx), 1e-6f));
                        ++idx;
                    }
                }
            }
        }
    }
}

TEST_CASE("Reshape Module - CF32", "[modules][reshape][CF32]") {
    auto implementations = Registry::ListAvailableModules("reshape");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("reshape", impl.device, impl.runtime, impl.provider);

            Modules::Reshape config;
            config.shape = "[2, 2]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({4});
            input.at(0) = {0.0f, 1.0f};
            input.at(1) = {2.0f, 3.0f};
            input.at(2) = {4.0f, 5.0f};
            input.at(3) = {6.0f, 7.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 2);

            REQUIRE_THAT(out.at<CF32>(0, 0).real(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 0).imag(), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 1).real(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 1).imag(), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1, 0).real(), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1, 0).imag(), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1, 1).real(), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1, 1).imag(), Catch::Matchers::WithinAbs(7.0f, 1e-6f));
        }
    }
}

TEST_CASE("Reshape Module - Size Mismatch Error", "[modules][reshape][error]") {
    auto implementations = Registry::ListAvailableModules("reshape");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireReshapeValidationError(impl, "[10]", {8});
        }
    }
}

TEST_CASE("Reshape Module - Validation rejects malformed shapes",
          "[modules][reshape][validation]") {
    auto implementations = Registry::ListAvailableModules("reshape");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("empty shape string") {
                RequireReshapeValidationError(impl, "", {4});
            }

            SECTION("missing shape brackets") {
                RequireReshapeValidationError(impl, "4,4", {16});
            }

            SECTION("shape with no dimensions") {
                RequireReshapeValidationError(impl, "[]", {4});
            }

            SECTION("shape with trailing comma") {
                RequireReshapeValidationError(impl, "[4,]", {4});
            }

            SECTION("shape with zero dimension") {
                RequireReshapeValidationError(impl, "[0,4]", {4});
            }

            SECTION("shape with no parseable dimensions") {
                RequireReshapeValidationError(impl, "[a,b]", {4});
            }

            SECTION("digit-containing malformed shape") {
                RequireReshapeValidationError(impl, "[2x, 2]", {4});
            }

            SECTION("dimension outside U64 range") {
                RequireReshapeValidationError(impl, "[18446744073709551616]", {4});
            }

            SECTION("target layout product overflow") {
                RequireReshapeValidationError(impl, "[18446744073709551615, 2]", {4});
            }
        }
    }
}

TEST_CASE("Reshape Module - Validation retains the original input layout",
          "[modules][reshape][validation][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("reshape");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor input;
            REQUIRE(input.create(impl.device, DataType::F32, {2, 4}) ==
                    Result::SUCCESS);

            TensorMap inputs;
            inputs["buffer"].requested("test", "buffer");
            inputs["buffer"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("reshape", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);

            Modules::Reshape config;
            config.shape = "[8]";
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Parser::Map update;
            update["shape"] = std::string("[4, 2]");
            REQUIRE(module->reconfigure(update, true) == Result::SUCCESS);
            REQUIRE(module->inputs().at("buffer").tensor.shape() == Shape{2, 4});
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
