#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <string>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/domains/core/slice/module.hh"

using namespace Jetstream;

namespace {

void RequireSliceValidationError(const Registry::ModuleRegistration& impl,
                                 const std::string& slice,
                                 const Shape& inputShape = {8}) {
    Tensor input;
    REQUIRE(input.create(impl.device, DataType::F32, inputShape) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    Modules::Slice config;
    config.slice = slice;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("slice", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);

    Result result = Result::SUCCESS;
    REQUIRE_NOTHROW(result = module->create("test", config, inputs));
    REQUIRE(result == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Slice plans bind to their source layout",
          "[modules][slice][validation][plan]") {
    Tensor input(DeviceType::CPU, DataType::F32, {4, 4});
    Tensor::SlicePlan plan;
    REQUIRE(input.planSlice({Token(1), Token()}, plan) == Result::SUCCESS);

    Tensor output = input.clone();
    REQUIRE(output.applySlicePlan(plan) == Result::SUCCESS);
    REQUIRE(output.shape() == Shape{4});
    REQUIRE(output.offset() == 4);

    Tensor stale = input.clone();
    REQUIRE(stale.slice({Token(1), Token()}) == Result::SUCCESS);
    REQUIRE(stale.applySlicePlan(plan) == Result::ERROR);
}

TEST_CASE("Slice Module - Basic Range F32", "[modules][slice][F32]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[2:6]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
        }
    }
}

TEST_CASE("Slice Module - Explicit Empty Range F32", "[modules][slice][F32][empty]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            for (const U64 start : {U64{0}, U64{1}}) {
                CAPTURE(start);
                TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

                Modules::Slice config;
                config.slice = "[" + std::to_string(start) + ":0]";
                ctx.setConfig(config);

                auto input = ctx.createTensor<F32>({4});
                ctx.setInput("buffer", input);
                REQUIRE(ctx.run() == Result::SUCCESS);

                const auto& out = ctx.output("buffer");
                REQUIRE(out.shape() == Shape{0});
                REQUIRE(out.size() == 0);
                REQUIRE(out.offset() == start);
                REQUIRE(out.contiguous());
            }
        }
    }
}

TEST_CASE("Slice Module - Step F32", "[modules][slice][F32][step]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[:8:2]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
        }
    }
}

TEST_CASE("Slice Module - Omitted Stop Step F32", "[modules][slice][F32][step]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[1::2]";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("buffer");
            REQUIRE(out.shape() == Shape{4});
            REQUIRE(out.offset() == 1);
            REQUIRE(out.stride() == Shape{2});
        }
    }
}

TEST_CASE("Slice Module - Single Index F32", "[modules][slice][F32][index]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[1]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 4});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // Selecting index 1 from a 2D tensor should reduce to 1D.
            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);
        }
    }
}

TEST_CASE("Slice Module - Ellipsis F32", "[modules][slice][F32][ellipsis]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[...]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 4});
            for (U64 i = 0; i < 16; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
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

TEST_CASE("Slice Module - 2D Slice F32", "[modules][slice][F32][2d]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[1:3, 0:2]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 4});
            for (U64 i = 0; i < 16; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 2);

            // Input was:
            // [[ 0,  1,  2,  3],
            //  [ 4,  5,  6,  7],
            //  [ 8,  9, 10, 11],
            //  [12, 13, 14, 15]]
            // After [1:3, 0:2]:
            // [[ 4,  5],
            //  [ 8,  9]]
            REQUIRE_THAT(out.at<F32>(0, 0), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(0, 1), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 0), Catch::Matchers::WithinAbs(8.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 1), Catch::Matchers::WithinAbs(9.0f, 1e-6f));
        }
    }
}

TEST_CASE("Slice Module - CF32", "[modules][slice][CF32]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[1:3]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({4});
            input.at(0) = {0.0f, 1.0f};
            input.at(1) = {2.0f, 3.0f};
            input.at(2) = {4.0f, 5.0f};
            input.at(3) = {6.0f, 7.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 2);

            REQUIRE_THAT(out.at<CF32>(0).real(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
        }
    }
}

TEST_CASE("Slice Module - Validation rejects malformed slice strings",
           "[modules][slice][validation]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("basic syntax failures") {
                for (const auto* malformed : {
                         "", "1:4", "[foo]", "[-1]", "[0,,1]", "[0 1]", "[[]]",
                         "[::]", "[1::]", "[1:2:]"}) {
                    CAPTURE(malformed);
                    RequireSliceValidationError(impl, malformed);
                }
            }

            SECTION("numeric overflow") {
                RequireSliceValidationError(impl, "[18446744073709551616:]");
                RequireSliceValidationError(impl, "[::18446744073709551616]");
            }

            SECTION("zero step") {
                RequireSliceValidationError(impl, "[::0]");
            }

            SECTION("duplicate ellipsis") {
                RequireSliceValidationError(impl, "[..., ...]", {2, 4});
            }

            SECTION("excess dimensions") {
                RequireSliceValidationError(impl, "[:, :]", {8});
            }

            SECTION("indices and ranges must fit their dimensions") {
                for (const auto* invalidRange : {"[8]", "[9:]", "[:9]"}) {
                    CAPTURE(invalidRange);
                    RequireSliceValidationError(impl, invalidRange, {8});
                }
            }
        }
    }
}

TEST_CASE("Slice Module - Direct creation preserves accepted syntax",
          "[modules][slice][validation][syntax]") {
    const auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor inputFixture;
            REQUIRE(inputFixture.create(impl.device, DataType::F32, {4, 4}) ==
                    Result::SUCCESS);

            for (const auto* syntax : {
                     "[]", "[ ]", "[...]", "[1]", "[:]", "[1:]", "[:3]",
                     "[:4]", "[4:]", "[1:3]", "[::2]", "[1::2]", "[:3:2]",
                     "[1:3:2]", "[1:0]", "[ 1:3 , ... ]"}) {
                CAPTURE(syntax);

                Tensor input = inputFixture.clone();
                TensorMap inputs;
                inputs["buffer"].requested("test", "buffer");
                inputs["buffer"].tensor = input;

                Modules::Slice config;
                config.slice = syntax;
                std::shared_ptr<Module> module;
                REQUIRE(Registry::BuildModule("slice", impl.device, impl.runtime,
                                              impl.provider, module) == Result::SUCCESS);

                Result result = Result::ERROR;
                REQUIRE_NOTHROW(result = module->create("test", config, inputs));
                REQUIRE(result == Result::SUCCESS);
                REQUIRE(module->outputs().contains("buffer"));
                REQUIRE(module->destroy() == Result::SUCCESS);
            }
        }
    }
}
