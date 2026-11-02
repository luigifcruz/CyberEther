#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <numbers>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/complex_merge/module.hh"

using namespace Jetstream;

namespace {

void RequireComplexMergeValidationError(const Registry::ModuleRegistration& impl,
                                        const Modules::ComplexMerge& config,
                                        const Tensor& inputA,
                                        const Tensor& inputB) {
    TensorMap inputs;
    inputs["a"].requested("test", "a");
    inputs["a"].tensor = inputA;
    inputs["b"].requested("test", "b");
    inputs["b"].tensor = inputB;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("complex_merge", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Complex Merge Module - Real and Imag", "[modules][complex_merge]") {
    auto implementations = Registry::ListAvailableModules("complex_merge");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("complex_merge", impl.device, impl.runtime, impl.provider);

            Modules::ComplexMerge config;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({3});
            auto b = ctx.createTensor<F32>({3});
            a.at(0) = 1.0f;
            a.at(1) = -0.5f;
            a.at(2) = 0.0f;
            b.at(0) = -2.0f;
            b.at(1) = 0.25f;
            b.at(2) = 3.0f;
            ctx.setInput("a", a);
            ctx.setInput("b", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("buffer");
            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE(out.shape() == Shape{3});
            REQUIRE(out.at<CF32>(0) == CF32(1.0f, -2.0f));
            REQUIRE(out.at<CF32>(1) == CF32(-0.5f, 0.25f));
            REQUIRE(out.at<CF32>(2) == CF32(0.0f, 3.0f));
        }
    }
}

TEST_CASE("Complex Merge Module - Magnitude and Phase", "[modules][complex_merge]") {
    auto implementations = Registry::ListAvailableModules("complex_merge");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("complex_merge", impl.device, impl.runtime, impl.provider);

            Modules::ComplexMerge config;
            config.mode = "polar";
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({2});
            auto b = ctx.createTensor<F32>({2});
            a.at(0) = 2.0f;
            a.at(1) = 5.0f;
            b.at(0) = std::numbers::pi_v<F32> / 2.0f;
            b.at(1) = 0.9272952f;
            ctx.setInput("a", a);
            ctx.setInput("b", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("buffer");
            REQUIRE_THAT(out.at<CF32>(0).real(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(), Catch::Matchers::WithinAbs(3.0f, 1e-5f));
            REQUIRE_THAT(out.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(4.0f, 1e-5f));
        }
    }
}

TEST_CASE("Complex Merge Module - Rejects invalid inputs and mode",
          "[modules][complex_merge][validation]") {
    auto implementations = Registry::ListAvailableModules("complex_merge");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor real8;
            REQUIRE(real8.create(impl.device, DataType::F32, {8}) == Result::SUCCESS);
            Tensor real4;
            REQUIRE(real4.create(impl.device, DataType::F32, {4}) == Result::SUCCESS);
            Tensor complex8;
            REQUIRE(complex8.create(impl.device, DataType::CF32, {8}) == Result::SUCCESS);

            RequireComplexMergeValidationError(impl, {}, real8, real4);
            RequireComplexMergeValidationError(impl, {}, real8, complex8);

            Modules::ComplexMerge config;
            config.mode = "cartesian";
            RequireComplexMergeValidationError(impl, config, real8, real8);

            Tensor scalar;
            REQUIRE(scalar.create(impl.device, DataType::F32, {1}) == Result::SUCCESS);
            REQUIRE(scalar.squeezeDims(0) == Result::SUCCESS);
            RequireComplexMergeValidationError(impl, {}, scalar, scalar);
        }
    }
}
