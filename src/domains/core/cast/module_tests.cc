#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/cast/module.hh"

using namespace Jetstream;

TEST_CASE("Cast Module - CI8 to CF32", "[modules][cast][CI8]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CI8>({4});
            input.at(0) = {64, -64};
            input.at(1) = {127, -128};
            input.at(2) = {0, 0};
            input.at(3) = {-1, 1};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);
            REQUIRE(out.dtype() == DataType::CF32);

            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(0.5f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(-0.5f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(1).real(),
                         Catch::Matchers::WithinAbs(127.0f / 128.0f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(1).imag(),
                         Catch::Matchers::WithinAbs(-1.0f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(2).real(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(2).imag(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Cast Module - F32 to F32", "[modules][cast][F32]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({3});
            input.at(0) = 0.5f;
            input.at(1) = -1.0f;
            input.at(2) = 0.0f;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(0.5f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(-1.0f, 1e-6f));
        }
    }
}

TEST_CASE("Cast Module - I8 to F32", "[modules][cast][I8]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<I8>({3});
            input.at(0) = 64;
            input.at(1) = -128;
            input.at(2) = 0;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(0.5f, 1e-3f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(-1.0f, 1e-3f));
        }
    }
}

TEST_CASE("Cast Module - U8 to F32", "[modules][cast][U8]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<U8>({3});
            input.at(0) = 128;
            input.at(1) = 255;
            input.at(2) = 0;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(1.0f, 1e-3f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(255.0f / 128.0f, 1e-3f));
        }
    }
}

TEST_CASE("Cast Module - CI16 to CF32", "[modules][cast][CI16]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CI16>({3});
            input.at(0) = {16384, -16384};
            input.at(1) = {32767, -32768};
            input.at(2) = {0, 0};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::CF32);

            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(0.5f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(-0.5f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(1).real(),
                         Catch::Matchers::WithinAbs(32767.0f / 32768.0f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(1).imag(),
                         Catch::Matchers::WithinAbs(-1.0f, 1e-3f));
        }
    }
}

TEST_CASE("Cast Module - I16 to F32", "[modules][cast][I16]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<I16>({3});
            input.at(0) = 16384;
            input.at(1) = -32768;
            input.at(2) = 0;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(0.5f, 1e-3f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(-1.0f, 1e-3f));
        }
    }
}

TEST_CASE("Cast Module - U16 to F32", "[modules][cast][U16]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<U16>({2});
            input.at(0) = 32768;
            input.at(1) = 65535;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(1.0f, 1e-3f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(65535.0f / 32768.0f, 1e-3f));
        }
    }
}

TEST_CASE("Cast Module - I32 to F32", "[modules][cast][I32]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<I32>({2});
            input.at(0) = 1073741824;
            input.at(1) = -2147483648;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(0.5f, 1e-3f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(-1.0f, 1e-3f));
        }
    }
}

TEST_CASE("Cast Module - U32 to F32", "[modules][cast][U32]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<U32>({2});
            input.at(0) = 2147483648U;
            input.at(1) = 0U;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(1.0f, 1e-3f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Cast Module - CI32 to CF32", "[modules][cast][CI32]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CI32>({2});
            input.at(0) = {1073741824, -1073741824};
            input.at(1) = {0, 0};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::CF32);

            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(0.5f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(-0.5f, 1e-3f));
        }
    }
}

TEST_CASE("Cast Module - CU8 to CF32", "[modules][cast][CU8]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CU8>({3});
            input.at(0) = {128, 0};
            input.at(1) = {255, 255};
            input.at(2) = {0, 0};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::CF32);

            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(1.0f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(),
                         Catch::Matchers::WithinAbs(255.0f / 128.0f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(2).real(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Cast Module - CU16 to CF32", "[modules][cast][CU16]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CU16>({2});
            input.at(0) = {32768, 0};
            input.at(1) = {0, 65535};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::CF32);

            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(1.0f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(),
                         Catch::Matchers::WithinAbs(65535.0f / 32768.0f, 1e-3f));
        }
    }
}

TEST_CASE("Cast Module - CU32 to CF32", "[modules][cast][CU32]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CU32>({2});
            input.at(0) = {2147483648U, 0};
            input.at(1) = {0, 0};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::CF32);

            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(1.0f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Cast Module - 2D Tensor CI8", "[modules][cast][CI8][2d]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CI8>({2, 3});
            input.at(0, 0) = {64, 0};
            input.at(0, 1) = {0, 64};
            input.at(0, 2) = {-128, 127};
            input.at(1, 0) = {0, 0};
            input.at(1, 1) = {1, -1};
            input.at(1, 2) = {-64, -64};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 3);

            REQUIRE_THAT(out.at<CF32>(0, 0).real(),
                         Catch::Matchers::WithinAbs(0.5f, 1e-3f));
            REQUIRE_THAT(out.at<CF32>(0, 0).imag(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Cast Module - Invalid Output Type", "[modules][cast][error]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "NOPE";
            ctx.setConfig(config);

            auto input = ctx.createTensor<CI8>({2});
            input.at(0) = {1, 2};
            input.at(1) = {3, 4};
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Cast Module rejects unsupported real to CF32 conversion",
          "[modules][cast][error][real]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "CF32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<I16>({2});
            input.at(0) = 1;
            input.at(1) = -1;
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Cast Module rejects unsupported complex to F32 conversion",
          "[modules][cast][error][complex]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<CI16>({2});
            input.at(0) = {1, 2};
            input.at(1) = {3, 4};
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Cast Module rejects unsupported CF32 to F32 conversion",
          "[modules][cast][error][cf32]") {
    auto implementations = Registry::ListAvailableModules("cast");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2});
            input.at(0) = {1.0f, 0.0f};
            input.at(1) = {0.0f, -1.0f};
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
