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

TEST_CASE("Multiply Module - Rank 4 Non-Contiguous F32",
          "[modules][multiply][F32][noncontiguous]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply", impl.device, impl.runtime, impl.provider);

            Tensor aStorage(DeviceType::CPU, DataType::F32, {2, 2, 3, 2, 4});
            Tensor bStorage(DeviceType::CPU, DataType::F32, {2, 1, 2, 1, 2});
            for (U64 i = 0; i < aStorage.size(); ++i) {
                aStorage.data<F32>()[i] = static_cast<F32>(i + 1);
            }
            for (U64 i = 0; i < bStorage.size(); ++i) {
                bStorage.data<F32>()[i] = static_cast<F32>(i + 1);
            }

            Tensor a = aStorage.clone();
            Tensor b = bStorage.clone();
            REQUIRE(a.slice({Token(1), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(b.slice({Token(1), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(a.permute({1, 0, 3, 2}) == Result::SUCCESS);
            REQUIRE(b.broadcastTo(a.shape()) == Result::SUCCESS);
            REQUIRE(a.shape() == Shape{3, 2, 4, 2});
            REQUIRE(a.offset() != 0);
            REQUIRE_FALSE(a.contiguous());
            REQUIRE_FALSE(b.contiguous());
            REQUIRE(b.stride(0) == 0);
            REQUIRE(b.stride(2) == 0);

            ctx.setInput("a", a);
            ctx.setInput("b", b);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("product");
            for (U64 i = 0; i < 3; ++i) {
                for (U64 j = 0; j < 2; ++j) {
                    for (U64 k = 0; k < 4; ++k) {
                        for (U64 l = 0; l < 2; ++l) {
                            const F32 expected = a.at<F32>(i, j, k, l) * b.at<F32>(i, j, k, l);
                            REQUIRE_THAT(out.at<F32>(i, j, k, l),
                                         Catch::Matchers::WithinAbs(expected, 1e-6f));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Multiply Module - Rank 4 Non-Contiguous CF32",
          "[modules][multiply][CF32][noncontiguous]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply", impl.device, impl.runtime, impl.provider);

            Tensor aStorage(DeviceType::CPU, DataType::CF32, {2, 2, 3, 2, 4});
            Tensor bStorage(DeviceType::CPU, DataType::CF32, {2, 1, 2, 1, 2});
            for (U64 i = 0; i < aStorage.size(); ++i) {
                const F32 value = static_cast<F32>(i + 1);
                aStorage.data<CF32>()[i] = CF32(value, value * 0.25f);
            }
            for (U64 i = 0; i < bStorage.size(); ++i) {
                const F32 value = static_cast<F32>(i + 1);
                bStorage.data<CF32>()[i] = CF32(value * 0.5f, -value * 0.125f);
            }

            Tensor a = aStorage.clone();
            Tensor b = bStorage.clone();
            REQUIRE(a.slice({Token(1), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(b.slice({Token(1), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(a.permute({1, 0, 3, 2}) == Result::SUCCESS);
            REQUIRE(b.broadcastTo(a.shape()) == Result::SUCCESS);
            REQUIRE(a.offset() != 0);
            REQUIRE_FALSE(a.contiguous());
            REQUIRE_FALSE(b.contiguous());

            ctx.setInput("a", a);
            ctx.setInput("b", b);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("product");
            for (U64 i = 0; i < 3; ++i) {
                for (U64 j = 0; j < 2; ++j) {
                    for (U64 k = 0; k < 4; ++k) {
                        for (U64 l = 0; l < 2; ++l) {
                            const CF32 expected = a.at<CF32>(i, j, k, l) * b.at<CF32>(i, j, k, l);
                            const CF32 actual = out.at<CF32>(i, j, k, l);
                            REQUIRE_THAT(actual.real(), Catch::Matchers::WithinAbs(expected.real(), 1e-5f));
                            REQUIRE_THAT(actual.imag(), Catch::Matchers::WithinAbs(expected.imag(), 1e-5f));
                        }
                    }
                }
            }
        }
    }
}
