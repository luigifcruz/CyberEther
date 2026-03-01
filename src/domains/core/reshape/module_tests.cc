#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/reshape/module.hh"

using namespace Jetstream;

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
            TestContext ctx("reshape", impl.device, impl.runtime, impl.provider);

            Modules::Reshape config;
            config.shape = "[10]";  // 10 != 8

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Reshape Module - Validation rejects malformed shapes",
          "[modules][reshape][validation]") {
    auto implementations = Registry::ListAvailableModules("reshape");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("empty shape string") {
            TestContext ctx("reshape", impl.device, impl.runtime,
                            impl.provider);
            Modules::Reshape config;
            config.shape = "";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("missing shape brackets") {
            TestContext ctx("reshape", impl.device, impl.runtime,
                            impl.provider);
            Modules::Reshape config;
            config.shape = "4,4";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({16});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("shape with zero dimension") {
            TestContext ctx("reshape", impl.device, impl.runtime,
                            impl.provider);
            Modules::Reshape config;
            config.shape = "[0,4]";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("shape with no parseable dimensions") {
            TestContext ctx("reshape", impl.device, impl.runtime,
                            impl.provider);
            Modules::Reshape config;
            config.shape = "[a,b]";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
