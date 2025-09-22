#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Multiply Module - F32", "[modules][multiply][F32]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply", impl.device, impl.runtime, impl.provider);

            auto a = ctx.createTensor<F32>({4});
            auto b = ctx.createTensor<F32>({4});
            a.at(0) = 1.0f;
            a.at(1) = 2.0f;
            a.at(2) = 3.0f;
            a.at(3) = 4.0f;
            b.at(0) = 2.0f;
            b.at(1) = 3.0f;
            b.at(2) = 4.0f;
            b.at(3) = 5.0f;

            ctx.setInput("a", a);
            ctx.setInput("b", b);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("product");
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(12.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(20.0f, 1e-6f));
        }
    }
}

TEST_CASE("Multiply Module - Broadcast Shape", "[modules][multiply][broadcast]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply", impl.device, impl.runtime, impl.provider);

            auto a = ctx.createTensor<F32>({2, 1});
            auto b = ctx.createTensor<F32>({2, 3});
            a.at(0, 0) = 2.0f;
            a.at(1, 0) = 3.0f;
            b.at(0, 0) = 1.0f;
            b.at(0, 1) = 2.0f;
            b.at(0, 2) = 3.0f;
            b.at(1, 0) = 4.0f;
            b.at(1, 1) = 5.0f;
            b.at(1, 2) = 6.0f;

            ctx.setInput("a", a);
            ctx.setInput("b", b);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("product");
            REQUIRE(out.rank() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 3);
            REQUIRE_THAT(out.at<F32>(1, 2), Catch::Matchers::WithinAbs(18.0f, 1e-6f));
        }
    }
}

TEST_CASE("Multiply Module - Non Broadcastable Shapes Error",
          "[modules][multiply][error]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply", impl.device, impl.runtime, impl.provider);

            auto a = ctx.createTensor<F32>({2, 3});
            auto b = ctx.createTensor<F32>({2, 2});

            ctx.setInput("a", a);
            ctx.setInput("b", b);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
