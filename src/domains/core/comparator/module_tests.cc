#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/comparator/module.hh"

using namespace Jetstream;

TEST_CASE("Comparator Module - F32 Equal", "[modules][comparator][F32]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            config.tolerance = 1e-6;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({4});
            auto b = ctx.createTensor<F32>({4});

            a.at(0) = 1.0f; a.at(1) = 2.0f; a.at(2) = 3.0f; a.at(3) = 4.0f;
            b.at(0) = 1.0f; b.at(1) = 2.0f; b.at(2) = 3.0f; b.at(3) = 4.0f;

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Comparator Module - F32 Unequal", "[modules][comparator][F32]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            config.tolerance = 1e-6;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({4});
            auto b = ctx.createTensor<F32>({4});

            a.at(0) = 1.0f; a.at(1) = 2.0f; a.at(2) = 3.0f; a.at(3) = 4.0f;
            b.at(0) = 1.0f; b.at(1) = 2.5f; b.at(2) = 3.0f; b.at(3) = 5.0f;

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(0.5f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
        }
    }
}

TEST_CASE("Comparator Module - CF32", "[modules][comparator][CF32]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            ctx.setConfig(config);

            auto a = ctx.createTensor<CF32>({2});
            auto b = ctx.createTensor<CF32>({2});

            a.at(0) = {3.0f, 4.0f};
            a.at(1) = {0.0f, 0.0f};
            b.at(0) = {0.0f, 0.0f};
            b.at(1) = {0.0f, 0.0f};

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(5.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Comparator Module - F64", "[modules][comparator][F64]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F64>({2});
            auto b = ctx.createTensor<F64>({2});

            a.at(0) = 1.0; a.at(1) = 2.0;
            b.at(0) = 1.5; b.at(1) = 2.0;

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE(out.dtype() == DataType::F64);
            REQUIRE_THAT(out.at<F64>(0), Catch::Matchers::WithinAbs(0.5, 1e-12));
            REQUIRE_THAT(out.at<F64>(1), Catch::Matchers::WithinAbs(0.0, 1e-12));
        }
    }
}

TEST_CASE("Comparator Module - CF64", "[modules][comparator][CF64]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            ctx.setConfig(config);

            auto a = ctx.createTensor<CF64>({1});
            auto b = ctx.createTensor<CF64>({1});

            a.at(0) = {0.0, 0.0};
            b.at(0) = {3.0, 4.0};

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE(out.dtype() == DataType::F64);
            REQUIRE_THAT(out.at<F64>(0), Catch::Matchers::WithinAbs(5.0, 1e-12));
        }
    }
}

TEST_CASE("Comparator Module - Three Inputs", "[modules][comparator][multi]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 3;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({2});
            auto b = ctx.createTensor<F32>({2});
            auto c = ctx.createTensor<F32>({2});

            a.at(0) = 0.0f; a.at(1) = 0.0f;
            b.at(0) = 1.0f; b.at(1) = 0.0f;
            c.at(0) = 0.0f; c.at(1) = 2.0f;

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);
            ctx.setInput("input2", c);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
        }
    }
}

TEST_CASE("Comparator Module - Shape Mismatch", "[modules][comparator][error]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({4});
            auto b = ctx.createTensor<F32>({3});

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Comparator Module - Dtype Mismatch", "[modules][comparator][error]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({4});
            auto b = ctx.createTensor<F64>({4});

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
