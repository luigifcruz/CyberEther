#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "jetstream/domains/dsp/squelch/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

void RequireSquelchValidationError(const Registry::ModuleRegistration& impl,
                                   const Modules::Squelch& config,
                                   const DataType dtype) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, {16}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("squelch", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Squelch Module - Supports F32 and CF32", "[modules][squelch]") {
    auto implementations = Registry::ListAvailableModules("squelch");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("F32") {
            TestContext ctx("squelch", impl.device, impl.runtime, impl.provider);
            Modules::Squelch config;
            config.threshold = 0.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({16});
            for (U64 i = 0; i < input.size(); ++i) {
                input.at(i) = 1.0f;
            }
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }

        SECTION("CF32") {
            TestContext ctx("squelch", impl.device, impl.runtime, impl.provider);
            Modules::Squelch config;
            config.threshold = 0.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({16});
            for (U64 i = 0; i < input.size(); ++i) {
                input.at(i) = CF32(1.0f, 0.0f);
            }
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Squelch Module - Rejects invalid candidates during validation",
          "[modules][squelch][validation]") {
    auto implementations = Registry::ListAvailableModules("squelch");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("threshold must be finite and non-negative") {
            Modules::Squelch config;
            config.threshold = -0.1f;
            RequireSquelchValidationError(impl, config, DataType::F32);

            config.threshold = std::numeric_limits<F32>::quiet_NaN();
            RequireSquelchValidationError(impl, config, DataType::F32);
        }

        SECTION("input dtype must be supported") {
            Modules::Squelch config;
            RequireSquelchValidationError(impl, config, DataType::U8);
        }
    }
}
