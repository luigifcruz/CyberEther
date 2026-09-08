#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <limits>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/cast/module.hh"

using namespace Jetstream;

namespace {

auto CastImplementations() {
    return Registry::ListAvailableModules("cast");
}

Tensor MakeCastTensor(const Registry::ModuleRegistration& impl,
                      const DataType dtype,
                      const Shape& shape,
                      const bool broadcast = false) {
    Tensor input;
    if (broadcast) {
        REQUIRE(input.create(impl.device, dtype, Shape(shape.size(), 1)) ==
                Result::SUCCESS);
        REQUIRE(input.broadcastTo(shape) == Result::SUCCESS);
    } else {
        REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    }
    return input;
}

void RequireCastValidationError(const Registry::ModuleRegistration& impl,
                                const Modules::Cast& config,
                                const Tensor& input) {
    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("cast", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Cast Module - CI8 to CF32", "[modules][cast][CI8]") {
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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

TEST_CASE("Cast Module - F32 to CF32", "[modules][cast][F32][CF32]") {
    auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "CF32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({3});
            input.at(0) = 0.5f;
            input.at(1) = -1.0f;
            input.at(2) = 0.0f;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(0.5f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(),
                         Catch::Matchers::WithinAbs(-1.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Cast Module - CF32 bypass", "[modules][cast][CF32][bypass]") {
    auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Modules::Cast config;
            config.outputType = "CF32";
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({3});
            input.at(0) = {1.0f, -2.0f};
            input.at(1) = {-3.0f, 4.0f};
            input.at(2) = {0.0f, 0.5f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(-2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(),
                         Catch::Matchers::WithinAbs(-3.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(),
                         Catch::Matchers::WithinAbs(4.0f, 1e-6f));
        }
    }
}

TEST_CASE("Cast Module preserves every matching dtype bypass",
          "[modules][cast][bypass]") {
    constexpr std::array<DataType, 20> kDataTypes = {
        DataType::F32, DataType::F64,
        DataType::I8, DataType::I16, DataType::I32, DataType::I64,
        DataType::U8, DataType::U16, DataType::U32, DataType::U64,
        DataType::CF32, DataType::CF64,
        DataType::CI8, DataType::CI16, DataType::CI32, DataType::CI64,
        DataType::CU8, DataType::CU16, DataType::CU32, DataType::CU64,
    };

    const auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const DataType dtype : kDataTypes) {
            DYNAMIC_SECTION("Type: " << dtype << " Device: " << impl.device
                            << " Runtime: " << impl.runtime) {
                const Tensor input = MakeCastTensor(impl, dtype, {1});
                TensorMap inputs;
                inputs["buffer"].requested("test", "buffer");
                inputs["buffer"].tensor = input;

                Modules::Cast config;
                config.outputType = std::string(DataTypeToName(dtype));

                std::shared_ptr<Module> module;
                REQUIRE(Registry::BuildModule("cast", impl.device, impl.runtime,
                                              impl.provider, module) == Result::SUCCESS);
                REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);
                REQUIRE(module->state() == Module::State::CREATED);
                REQUIRE(module->outputs().at("buffer").tensor.dtype() == dtype);
                REQUIRE(module->outputs().at("buffer").tensor.id() == input.id());
            }
        }
    }
}

TEST_CASE("Cast Module - I8 to F32", "[modules][cast][I8]") {
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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
    auto implementations = CastImplementations();
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

TEST_CASE("Cast Module - Non-Contiguous I8 to F32",
          "[modules][cast][I8][noncontiguous]") {
    const auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("cast", impl.device, impl.runtime, impl.provider);

            Tensor storage(DeviceType::CPU, DataType::I8, {2, 3, 4});
            for (U64 index = 0; index < storage.size(); ++index) {
                storage.data<I8>()[index] =
                    static_cast<I8>(static_cast<I64>(index) - 12);
            }

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(1), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(input.permute({1, 0}) == Result::SUCCESS);
            REQUIRE(input.shape() == Shape{4, 3});
            REQUIRE(input.offset() != 0);
            REQUIRE_FALSE(input.contiguous());

            Modules::Cast config;
            config.outputType = "F32";
            ctx.setConfig(config);
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const Tensor& output = ctx.output("buffer");
            REQUIRE(output.shape() == input.shape());
            for (U64 row = 0; row < output.shape(0); ++row) {
                for (U64 column = 0; column < output.shape(1); ++column) {
                    const F32 expected = static_cast<F32>(input.at<I8>(row, column)) /
                                         128.0f;
                    REQUIRE_THAT(output.at<F32>(row, column),
                                 Catch::Matchers::WithinAbs(expected, 1e-6f));
                }
            }
        }
    }
}

TEST_CASE("Cast Module validation rejects invalid output spelling",
          "[modules][cast][validation][config]") {
    const auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const char* spelling : {"", "cf32", "CF32 ", "NONE", "NOPE"}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime
                            << " Output: " << spelling) {
                Modules::Cast config;
                config.outputType = spelling;
                const Tensor input = MakeCastTensor(impl, DataType::CI8, {2});
                RequireCastValidationError(impl, config, input);
            }
        }
    }
}

TEST_CASE("Cast Module provider validation rejects unsupported pairs",
          "[modules][cast][validation][pair]") {
    struct InvalidPair {
        DataType input;
        const char* output;
    };
    constexpr std::array<InvalidPair, 7> kInvalidPairs = {{
        {DataType::CF32, "F32"},
        {DataType::I16, "CF32"},
        {DataType::CI16, "F32"},
        {DataType::I64, "F32"},
        {DataType::U64, "F32"},
        {DataType::CI64, "CF32"},
        {DataType::CU64, "CF32"},
    }};

    const auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const auto& pair : kInvalidPairs) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime
                            << " Pair: " << pair.input << " -> " << pair.output) {
                Modules::Cast config;
                config.outputType = pair.output;
                const Tensor input = MakeCastTensor(impl, pair.input, {2});
                RequireCastValidationError(impl, config, input);
            }
        }
    }
}

TEST_CASE("Cast Module validation rejects logical output byte overflow",
          "[modules][cast][validation][overflow]") {
    const auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Cast config;
            config.outputType = "F32";
            const Shape shape = {
                std::numeric_limits<U64>::max() / sizeof(F32) + 1,
            };
            const Tensor input = MakeCastTensor(impl, DataType::I8, shape, true);
            RequireCastValidationError(impl, config, input);
        }
    }
}

TEST_CASE("Cast Module validation rejects non-bypass rank-zero output",
          "[modules][cast][validation][rank]") {
    const auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor input = MakeCastTensor(impl, DataType::I8, {1});
            REQUIRE(input.squeezeDims(0) == Result::SUCCESS);

            Modules::Cast config;
            config.outputType = "F32";
            RequireCastValidationError(impl, config, input);
        }
    }
}

TEST_CASE("Cast Module CPU validation rejects unsupported allocation size",
          "[modules][cast][validation][allocation]") {
    const auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Cast config;
            config.outputType = "F32";
            const Shape shape = {std::numeric_limits<U64>::max() / sizeof(F32)};
            const Tensor input = MakeCastTensor(impl, DataType::I8, shape, true);
            RequireCastValidationError(impl, config, input);
        }
    }
}

TEST_CASE("Cast Module CUDA validation rejects unsupported grid size",
          "[modules][cast][validation][cuda]") {
    const auto implementations = CastImplementations();
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CUDA) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Cast config;
            config.outputType = "F32";
            const Shape shape = {
                static_cast<U64>(std::numeric_limits<I32>::max()) * 256 + 1,
            };
            const Tensor input = MakeCastTensor(impl, DataType::I8, shape, true);
            RequireCastValidationError(impl, config, input);
        }
    }
}
