#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <numbers>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/complex_split/module.hh"

using namespace Jetstream;

namespace {

void RequireComplexSplitValidationError(const Registry::ModuleRegistration& impl,
                                        const Modules::ComplexSplit& config,
                                        const Tensor& input) {
    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("complex_split", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Complex Split Module - Real and Imag", "[modules][complex_split]") {
    auto implementations = Registry::ListAvailableModules("complex_split");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("complex_split", impl.device, impl.runtime, impl.provider);

            Modules::ComplexSplit config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2, 2});
            input.at(0, 0) = {1.0f, -2.0f};
            input.at(0, 1) = {0.5f, 0.25f};
            input.at(1, 0) = {0.0f, 0.0f};
            input.at(1, 1) = {-3.0f, 4.0f};
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& a = ctx.output("a");
            const auto& b = ctx.output("b");
            REQUIRE(a.dtype() == DataType::F32);
            REQUIRE(b.dtype() == DataType::F32);
            REQUIRE(a.shape() == Shape{2, 2});
            REQUIRE(b.shape() == Shape{2, 2});

            REQUIRE(a.at<F32>(0, 0) == 1.0f);
            REQUIRE(b.at<F32>(0, 0) == -2.0f);
            REQUIRE(a.at<F32>(0, 1) == 0.5f);
            REQUIRE(b.at<F32>(0, 1) == 0.25f);
            REQUIRE(a.at<F32>(1, 1) == -3.0f);
            REQUIRE(b.at<F32>(1, 1) == 4.0f);
        }
    }
}

TEST_CASE("Complex Split Module - Magnitude and Phase", "[modules][complex_split]") {
    auto implementations = Registry::ListAvailableModules("complex_split");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("complex_split", impl.device, impl.runtime, impl.provider);

            Modules::ComplexSplit config;
            config.mode = "polar";
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({3});
            input.at(0) = {3.0f, 4.0f};
            input.at(1) = {0.0f, -2.0f};
            input.at(2) = {-1.0f, 0.0f};
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& a = ctx.output("a");
            const auto& b = ctx.output("b");
            REQUIRE_THAT(a.at<F32>(0), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(b.at<F32>(0), Catch::Matchers::WithinAbs(0.9272952f, 1e-6f));
            REQUIRE_THAT(a.at<F32>(1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(b.at<F32>(1),
                         Catch::Matchers::WithinAbs(-std::numbers::pi_v<F32> / 2.0f, 1e-6f));
            REQUIRE_THAT(a.at<F32>(2), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(b.at<F32>(2),
                         Catch::Matchers::WithinAbs(std::numbers::pi_v<F32>, 1e-6f));
        }
    }
}

TEST_CASE("Complex Split Module - Strided input", "[modules][complex_split]") {
    auto implementations = Registry::ListAvailableModules("complex_split");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("complex_split", impl.device, impl.runtime, impl.provider);

            Modules::ComplexSplit config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({1});
            input.at(0) = {2.0f, -1.0f};
            REQUIRE(input.broadcastTo({4}) == Result::SUCCESS);
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& a = ctx.output("a");
            const auto& b = ctx.output("b");
            REQUIRE(a.shape() == Shape{4});
            for (U64 i = 0; i < 4; ++i) {
                REQUIRE(a.at<F32>(i) == 2.0f);
                REQUIRE(b.at<F32>(i) == -1.0f);
            }
        }
    }
}

TEST_CASE("Complex Split Module - Rejects invalid input and mode",
          "[modules][complex_split][validation]") {
    auto implementations = Registry::ListAvailableModules("complex_split");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor real;
            REQUIRE(real.create(impl.device, DataType::F32, {8}) == Result::SUCCESS);
            RequireComplexSplitValidationError(impl, {}, real);

            Tensor complex;
            REQUIRE(complex.create(impl.device, DataType::CF32, {8}) == Result::SUCCESS);
            Modules::ComplexSplit config;
            config.mode = "cartesian";
            RequireComplexSplitValidationError(impl, config, complex);

            Tensor scalar;
            REQUIRE(scalar.create(impl.device, DataType::CF32, {1}) == Result::SUCCESS);
            REQUIRE(scalar.squeezeDims(0) == Result::SUCCESS);
            RequireComplexSplitValidationError(impl, {}, scalar);
        }
    }
}
