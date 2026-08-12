#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <any>
#include <limits>
#include <optional>

#include "jetstream/domains/core/multiply/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

Tensor MakeMultiplyTensor(const Registry::ModuleRegistration& impl,
                          const DataType dtype,
                          const Shape& shape,
                          const bool broadcast = false) {
    Tensor tensor;
    if (broadcast) {
        REQUIRE(tensor.create(impl.device, dtype, Shape(shape.size(), 1)) == Result::SUCCESS);
        REQUIRE(tensor.broadcastTo(shape) == Result::SUCCESS);
    } else {
        REQUIRE(tensor.create(impl.device, dtype, shape) == Result::SUCCESS);
    }
    return tensor;
}

void RequireMultiplyValidationError(const Registry::ModuleRegistration& impl,
                                    const Tensor& tensorA,
                                    const Tensor& tensorB) {
    TensorMap inputs;
    inputs["a"].requested("test", "a");
    inputs["a"].tensor = tensorA;
    inputs["b"].requested("test", "b");
    inputs["b"].tensor = tensorB;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("multiply", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::Multiply{}, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

void RequireMultiplySignalAxes(const Registry::ModuleRegistration& impl,
                               const Shape& shapeA,
                               const SignalAxes& axesA,
                               const Shape& shapeB,
                               const SignalAxes& axesB,
                               const SignalAxes& expectedAxes) {
    TestContext ctx("multiply", impl.device, impl.runtime, impl.provider);
    auto a = ctx.createTensor<F32>(shapeA);
    auto b = ctx.createTensor<F32>(shapeB);
    for (U64 i = 0; i < a.size(); ++i) {
        a.data()[i] = 2.0f;
    }
    for (U64 i = 0; i < b.size(); ++i) {
        b.data()[i] = 3.0f;
    }
    if (axesA.sample || axesA.batch || axesA.channel) {
        REQUIRE(SetSignalAxes(a, axesA) == Result::SUCCESS);
    }
    if (axesB.sample || axesB.batch || axesB.channel) {
        REQUIRE(SetSignalAxes(b, axesB) == Result::SUCCESS);
    }

    ctx.setInput("a", a);
    ctx.setInput("b", b);
    REQUIRE(ctx.run() == Result::SUCCESS);

    const auto& out = ctx.output("product");
    SignalAxes outputAxes;
    REQUIRE(ResolveSignalAxes(out, outputAxes) == Result::SUCCESS);
    REQUIRE(outputAxes.sample == expectedAxes.sample);
    REQUIRE(outputAxes.batch == expectedAxes.batch);
    REQUIRE(outputAxes.channel == expectedAxes.channel);
}

}  // namespace

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
            REQUIRE(a.shape() == Shape{2, 1});
            REQUIRE(b.shape() == Shape{2, 3});

            auto& out = ctx.output("product");
            REQUIRE(out.rank() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 3);
            REQUIRE_THAT(out.at<F32>(1, 2), Catch::Matchers::WithinAbs(18.0f, 1e-6f));
        }
    }
}

TEST_CASE("Multiply Module - Merges Broadcast Signal Axes",
          "[modules][multiply][broadcast][metadata]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("roles from input a are right aligned") {
                RequireMultiplySignalAxes(impl, {1}, {.sample = Index{0}},
                                          {1, 1, 1}, {},
                                          {.sample = Index{2}});
            }

            SECTION("roles from input b are right aligned") {
                RequireMultiplySignalAxes(
                    impl, {1, 1, 1}, {},
                    {1, 1}, {.sample = Index{1}, .channel = Index{0}},
                    {.sample = Index{2}, .channel = Index{1}});
            }

            SECTION("matching roles merge from both inputs") {
                RequireMultiplySignalAxes(
                    impl,
                    {1, 1, 1},
                    {.sample = Index{2}, .batch = Index{0}, .channel = Index{1}},
                    {1, 1},
                    {.sample = Index{1}, .channel = Index{0}},
                    {.sample = Index{2}, .batch = Index{0}, .channel = Index{1}});
            }

            SECTION("same roles mapped to different axes conflict") {
                Tensor a;
                Tensor b;
                REQUIRE(a.create(impl.device, DataType::F32, {1, 1}) == Result::SUCCESS);
                REQUIRE(b.create(impl.device, DataType::F32, {1, 1, 1}) == Result::SUCCESS);
                REQUIRE(SetSignalAxes(a, {.sample = Index{0}}) == Result::SUCCESS);
                REQUIRE(SetSignalAxes(b, {.sample = Index{0}}) == Result::SUCCESS);
                RequireMultiplyValidationError(impl, a, b);
            }

            SECTION("different roles mapped to the same axis conflict") {
                Tensor a;
                Tensor b;
                REQUIRE(a.create(impl.device, DataType::F32, {1, 1}) == Result::SUCCESS);
                REQUIRE(b.create(impl.device, DataType::F32, {1, 1, 1}) == Result::SUCCESS);
                REQUIRE(SetSignalAxes(
                    a, {.sample = Index{1}, .channel = Index{0}}) == Result::SUCCESS);
                REQUIRE(SetSignalAxes(
                    b, {.sample = Index{2}, .batch = Index{1}}) == Result::SUCCESS);
                RequireMultiplyValidationError(impl, a, b);
            }

            SECTION("malformed signal metadata") {
                Tensor wrongTypeA;
                Tensor wrongTypeB;
                REQUIRE(wrongTypeA.create(impl.device, DataType::F32, {1, 1}) ==
                        Result::SUCCESS);
                REQUIRE(wrongTypeB.create(impl.device, DataType::F32, {1, 1}) ==
                        Result::SUCCESS);
                REQUIRE(wrongTypeA.setAttribute(
                    std::string(SampleAxisAttribute), Index{1}) == Result::SUCCESS);
                REQUIRE(wrongTypeA.setAttribute(
                    std::string(ChannelAxisAttribute), I64{0}) == Result::SUCCESS);
                RequireMultiplyValidationError(impl, wrongTypeA, wrongTypeB);

                Tensor outOfRangeA;
                Tensor outOfRangeB;
                REQUIRE(outOfRangeA.create(impl.device, DataType::F32, {1}) == Result::SUCCESS);
                REQUIRE(outOfRangeB.create(impl.device, DataType::F32, {1}) == Result::SUCCESS);
                REQUIRE(outOfRangeB.setAttribute(
                    std::string(SampleAxisAttribute), Index{0}) == Result::SUCCESS);
                REQUIRE(outOfRangeB.setAttribute(
                    std::string(ChannelAxisAttribute), Index{1}) == Result::SUCCESS);
                RequireMultiplyValidationError(impl, outOfRangeA, outOfRangeB);
            }
        }
    }
}

TEST_CASE("Multiply Module - Non Broadcastable Shapes Error",
          "[modules][multiply][error]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const Tensor a = MakeMultiplyTensor(impl, DataType::F32, {2, 3});
            const Tensor b = MakeMultiplyTensor(impl, DataType::F32, {2, 2});
            RequireMultiplyValidationError(impl, a, b);
        }
    }
}

TEST_CASE("Multiply Module - Validation rejects dtype mismatch and unsupported dtype",
          "[modules][multiply][validation][dtype]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const Tensor f32 = MakeMultiplyTensor(impl, DataType::F32, {4});
            const Tensor cf32 = MakeMultiplyTensor(impl, DataType::CF32, {4});
            RequireMultiplyValidationError(impl, f32, cf32);

            const Tensor f64A = MakeMultiplyTensor(impl, DataType::F64, {4});
            const Tensor f64B = MakeMultiplyTensor(impl, DataType::F64, {4});
            RequireMultiplyValidationError(impl, f64A, f64B);
        }
    }
}

TEST_CASE("Multiply Module - Validation rejects output layout overflow",
          "[modules][multiply][validation][overflow]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    constexpr U64 kLargeDimension = U64{1} << 32;
    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const Tensor a = MakeMultiplyTensor(impl, DataType::F32,
                                                {kLargeDimension, 1}, true);
            const Tensor b = MakeMultiplyTensor(impl, DataType::F32,
                                                {1, kLargeDimension}, true);
            RequireMultiplyValidationError(impl, a, b);

            const Tensor byteA = MakeMultiplyTensor(impl, DataType::F32,
                                                    {kLargeDimension, 1}, true);
            const Tensor byteB = MakeMultiplyTensor(impl, DataType::F32,
                                                    {1, U64{1} << 30}, true);
            RequireMultiplyValidationError(impl, byteA, byteB);
        }
    }
}

TEST_CASE("Multiply Module - CPU validation rejects unsupported allocation size",
          "[modules][multiply][validation][allocation]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const Shape shape = {std::numeric_limits<U64>::max() / sizeof(F32)};
            const Tensor a = MakeMultiplyTensor(impl, DataType::F32, shape, true);
            const Tensor b = MakeMultiplyTensor(impl, DataType::F32, {1});
            RequireMultiplyValidationError(impl, a, b);
        }
    }
}

TEST_CASE("Multiply Module - CUDA validation rejects unsupported grid size",
          "[modules][multiply][validation][cuda]") {
    const auto implementations = Registry::ListAvailableModules("multiply");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CUDA) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const Shape shape = {
                static_cast<U64>(std::numeric_limits<I32>::max()) * 256 + 1,
            };
            const Tensor a = MakeMultiplyTensor(impl, DataType::F32, shape, true);
            const Tensor b = MakeMultiplyTensor(impl, DataType::F32, {1});
            RequireMultiplyValidationError(impl, a, b);
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
