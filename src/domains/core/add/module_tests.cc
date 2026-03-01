#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("Add Module - F32", "[modules][add][F32]") {
    auto implementations = Registry::ListAvailableModules("add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("add", impl.device, impl.runtime, impl.provider);

            auto a = ctx.createTensor<F32>({4});
            auto b = ctx.createTensor<F32>({4});

            a.at(0) = 1.0f; a.at(1) = 2.0f; a.at(2) = 3.0f; a.at(3) = 4.0f;
            b.at(0) = 5.0f; b.at(1) = 6.0f; b.at(2) = 7.0f; b.at(3) = 8.0f;

            ctx.setInput("a", a);
            ctx.setInput("b", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("sum");
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(8.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(10.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(12.0f, 1e-6f));
        }
    }
}

TEST_CASE("Add Module - CF32", "[modules][add][CF32]") {
    auto implementations = Registry::ListAvailableModules("add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("add", impl.device, impl.runtime, impl.provider);

            auto a = ctx.createTensor<CF32>({2});
            auto b = ctx.createTensor<CF32>({2});

            a.at(0) = {1.0f, 2.0f}; a.at(1) = {3.0f, 4.0f};
            b.at(0) = {5.0f, 6.0f}; b.at(1) = {7.0f, 8.0f};

            ctx.setInput("a", a);
            ctx.setInput("b", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("sum");
            REQUIRE_THAT(out.at<CF32>(0).real(), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(8.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(), Catch::Matchers::WithinAbs(10.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(12.0f, 1e-6f));
        }
    }
}

TEST_CASE("Add Module - Broadcast F32", "[modules][add][broadcast]") {
    auto implementations = Registry::ListAvailableModules("add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("add", impl.device, impl.runtime, impl.provider);

            auto a = ctx.createTensor<F32>({2, 1});
            auto b = ctx.createTensor<F32>({2, 3});

            a.at(0, 0) = 1.0f;
            a.at(1, 0) = 2.0f;
            b.at(0, 0) = 10.0f;
            b.at(0, 1) = 20.0f;
            b.at(0, 2) = 30.0f;
            b.at(1, 0) = 40.0f;
            b.at(1, 1) = 50.0f;
            b.at(1, 2) = 60.0f;

            ctx.setInput("a", a);
            ctx.setInput("b", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("sum");
            REQUIRE(out.rank() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 3);
            REQUIRE_THAT(out.at<F32>(1, 2), Catch::Matchers::WithinAbs(62.0f, 1e-6f));
        }
    }
}

TEST_CASE("Add Module - Non Broadcastable Shapes Error", "[modules][add][error]") {
    auto implementations = Registry::ListAvailableModules("add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("add", impl.device, impl.runtime, impl.provider);

            auto a = ctx.createTensor<F32>({2, 3});
            auto b = ctx.createTensor<F32>({2, 2});

            ctx.setInput("a", a);
            ctx.setInput("b", b);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
