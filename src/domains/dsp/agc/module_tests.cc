#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>

#include "jetstream/domains/dsp/agc/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

void RequireAgcValidationError(const Registry::ModuleRegistration& impl) {
    Tensor input;
    REQUIRE(input.create(impl.device, DataType::U8, {16}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("agc", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    Modules::Agc config;
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("AGC - Normalizes CF32 peak", "[modules][agc][cf32]") {
    auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            ctx.setConfig(Modules::Agc{});

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {32}) == Result::SUCCESS);

            for (U64 i = 0; i < 32; ++i) {
                input.at<CF32>(i) = CF32(0.25f * static_cast<F32>(i + 1), -0.1f);
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE(out.shape(0) == 32);

            F32 peak = 0.0f;
            for (U64 i = 0; i < 32; ++i) {
                peak = std::max(peak, std::abs(out.at<CF32>(i)));
            }

            REQUIRE_THAT(peak, Catch::Matchers::WithinAbs(1.0f, 1e-5f));
        }
    }
}

TEST_CASE("AGC - Normalizes F32 peak", "[modules][agc][f32]") {
    auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            ctx.setConfig(Modules::Agc{});

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {16}) == Result::SUCCESS);

            for (U64 i = 0; i < 16; ++i) {
                input.at<F32>(i) = static_cast<F32>(i) - 8.0f;
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE(out.shape(0) == 16);

            F32 peak = 0.0f;
            for (U64 i = 0; i < 16; ++i) {
                peak = std::max(peak, std::abs(out.at<F32>(i)));
            }

            REQUIRE_THAT(peak, Catch::Matchers::WithinAbs(1.0f, 1e-5f));
        }
    }
}

TEST_CASE("AGC - Normalizes sample lanes independently",
          "[modules][agc][f32][metadata]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            ctx.setConfig(Modules::Agc{});

            Tensor input(DeviceType::CPU, DataType::F32, {2, 3});
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .channel = Index{0},
            }) == Result::SUCCESS);
            input.at<F32>(0, 0) = 1.0f;
            input.at<F32>(0, 1) = 2.0f;
            input.at<F32>(0, 2) = 4.0f;
            input.at<F32>(1, 0) = 10.0f;
            input.at<F32>(1, 1) = 20.0f;
            input.at<F32>(1, 2) = 20.0f;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE_THAT(out.at<F32>(0, 2),
                         Catch::Matchers::WithinAbs(1.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(1, 1),
                         Catch::Matchers::WithinAbs(1.0f, 1e-5f));
        }
    }
}

TEST_CASE("AGC - Rejects unsupported dtype", "[modules][agc][validation]") {
    auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireAgcValidationError(impl);
        }
    }
}
