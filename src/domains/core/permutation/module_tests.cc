#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/permutation/module.hh"

using namespace Jetstream;

TEST_CASE("Permutation Module - Transpose F32", "[modules][permutation][F32]") {
    auto implementations = Registry::ListAvailableModules("permutation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("permutation", impl.device, impl.runtime, impl.provider);

            Modules::Permutation config;
            config.permutation = {1, 0};
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            for (U64 i = 0; i < 6; ++i) {
                input.at(i / 3, i % 3) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.rank() == 2);
            REQUIRE(out.shape(0) == 3);
            REQUIRE(out.shape(1) == 2);
            REQUIRE(out.stride(0) == 1);
            REQUIRE(out.stride(1) == 3);
            REQUIRE(!out.contiguous());

            REQUIRE_THAT(out.at<F32>(0, 0), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(0, 1), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 1), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2, 0), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2, 1), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
        }
    }
}

TEST_CASE("Permutation Module - Reorder 3D CF32", "[modules][permutation][CF32]") {
    auto implementations = Registry::ListAvailableModules("permutation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("permutation", impl.device, impl.runtime, impl.provider);

            Modules::Permutation config;
            config.permutation = {2, 0, 1};
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2, 3, 4});
            U64 index = 0;
            for (U64 i = 0; i < 2; ++i) {
                for (U64 j = 0; j < 3; ++j) {
                    for (U64 k = 0; k < 4; ++k) {
                        input.at(i, j, k) = {static_cast<F32>(index), static_cast<F32>(index + 100)};
                        ++index;
                    }
                }
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.rank() == 3);
            REQUIRE(out.shape(0) == 4);
            REQUIRE(out.shape(1) == 2);
            REQUIRE(out.shape(2) == 3);
            REQUIRE(!out.contiguous());

            REQUIRE_THAT(out.at<CF32>(0, 0, 0).real(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 0, 0).imag(), Catch::Matchers::WithinAbs(100.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(3, 1, 2).real(), Catch::Matchers::WithinAbs(23.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(3, 1, 2).imag(), Catch::Matchers::WithinAbs(123.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1, 1, 0).real(), Catch::Matchers::WithinAbs(13.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1, 1, 0).imag(), Catch::Matchers::WithinAbs(113.0f, 1e-6f));
        }
    }
}

TEST_CASE("Permutation Module - Identity Permutation F32", "[modules][permutation][F32][identity]") {
    auto implementations = Registry::ListAvailableModules("permutation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("permutation", impl.device, impl.runtime, impl.provider);

            Modules::Permutation config;
            config.permutation = {0, 1};
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            for (U64 i = 0; i < 6; ++i) {
                input.at(i / 3, i % 3) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 3);
            REQUIRE(out.contiguous());
            REQUIRE_THAT(out.at<F32>(1, 2), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
        }
    }
}

TEST_CASE("Permutation Module - Transpose I32", "[modules][permutation][I32]") {
    auto implementations = Registry::ListAvailableModules("permutation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("permutation", impl.device, impl.runtime, impl.provider);

            Modules::Permutation config;
            config.permutation = {1, 0};
            ctx.setConfig(config);

            auto input = ctx.createTensor<I32>({2, 3});
            for (U64 i = 0; i < 6; ++i) {
                input.at(i / 3, i % 3) = static_cast<I32>(i + 10);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.rank() == 2);
            REQUIRE(out.shape(0) == 3);
            REQUIRE(out.shape(1) == 2);
            REQUIRE(out.stride(0) == 1);
            REQUIRE(out.stride(1) == 3);
            REQUIRE(!out.contiguous());

            REQUIRE(out.at<I32>(0, 0) == 10);
            REQUIRE(out.at<I32>(0, 1) == 13);
            REQUIRE(out.at<I32>(1, 0) == 11);
            REQUIRE(out.at<I32>(2, 1) == 15);
        }
    }
}

TEST_CASE("Permutation Module - Validation rejects invalid permutations",
          "[modules][permutation][validation]") {
    auto implementations = Registry::ListAvailableModules("permutation");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("empty permutation") {
            TestContext ctx("permutation", impl.device, impl.runtime, impl.provider);

            Modules::Permutation config;
            config.permutation = {};
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("duplicate axis") {
            TestContext ctx("permutation", impl.device, impl.runtime, impl.provider);

            Modules::Permutation config;
            config.permutation = {1, 1};
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("axis out of range") {
            TestContext ctx("permutation", impl.device, impl.runtime, impl.provider);

            Modules::Permutation config;
            config.permutation = {0, 2};
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("rank mismatch") {
            TestContext ctx("permutation", impl.device, impl.runtime, impl.provider);

            Modules::Permutation config;
            config.permutation = {1, 0};
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3, 4});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
