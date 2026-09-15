#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <limits>

#include "jetstream/domains/core/remove_indices/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Remove Indices Module - Telescope antenna selection",
          "[modules][remove_indices]") {
    const auto implementations = Registry::ListAvailableModules("remove_indices");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("remove_indices", impl.device, impl.runtime, impl.provider);
            Modules::RemoveIndices config;
            config.axis = 0;
            config.indices = {19, 2, 7, 2};
            ctx.setConfig(config);

            // Retain the packet layout with smaller channel/time dimensions.
            auto input = ctx.createTensor<CF32>({28, 3, 5, 2});
            for (U64 i = 0; i < input.size(); ++i) {
                input.data()[i] = {static_cast<F32>(i), -static_cast<F32>(i)};
            }
            REQUIRE(SetSignalAxes(input, {.sample = 2, .batch = 0, .channel = 1}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("packet", U64{42}) == Result::SUCCESS);
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("buffer");
            REQUIRE(out.shape() == Shape{25, 3, 5, 2});
            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE(out.contiguous());
            REQUIRE(std::any_cast<U64>(out.attribute("packet")) == 42);
            SignalAxes axes;
            REQUIRE(ResolveSignalAxes(out, axes) == Result::SUCCESS);
            REQUIRE(axes.sample == 2);
            REQUIRE(axes.batch == 0);
            REQUIRE(axes.channel == 1);

            U64 outputAntenna = 0;
            for (U64 antenna = 0; antenna < 28; ++antenna) {
                if (antenna == 2 || antenna == 7 || antenna == 19) {
                    continue;
                }
                for (U64 channel = 0; channel < 3; ++channel) {
                    for (U64 time = 0; time < 5; ++time) {
                        for (U64 polarization = 0; polarization < 2; ++polarization) {
                            REQUIRE(out.at<CF32>(outputAntenna, channel, time, polarization) ==
                                    input.at(antenna, channel, time, polarization));
                        }
                    }
                }
                ++outputAntenna;
            }
        }
    }
}

TEST_CASE("Remove Indices Module - Every positive and negative axis",
          "[modules][remove_indices][axis]") {
    const auto implementations = Registry::ListAvailableModules("remove_indices");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (I64 axis = -4; axis < 4; ++axis) {
            DYNAMIC_SECTION("Device: " << impl.device << " Axis: " << axis) {
                TestContext ctx("remove_indices", impl.device, impl.runtime, impl.provider);
                const Index resolvedAxis = axis < 0 ? axis + 4 : axis;
                const Shape shape = {4, 5, 6, 3};
                Modules::RemoveIndices config;
                config.axis = axis;
                config.indices = {shape[resolvedAxis] - 1, 1, 1};
                ctx.setConfig(config);

                auto input = ctx.createTensor<I32>(shape);
                for (U64 i = 0; i < input.size(); ++i) {
                    input.data()[i] = static_cast<I32>(i);
                }
                ctx.setInput("buffer", input);
                REQUIRE(ctx.run() == Result::SUCCESS);

                auto expectedShape = shape;
                expectedShape[resolvedAxis] -= 2;
                const auto& out = ctx.output("buffer");
                REQUIRE(out.shape() == expectedShape);
                REQUIRE(out.contiguous());

                U64 outputIndex = 0;
                for (U64 a = 0; a < shape[0]; ++a) {
                    for (U64 c = 0; c < shape[1]; ++c) {
                        for (U64 t = 0; t < shape[2]; ++t) {
                            for (U64 p = 0; p < shape[3]; ++p) {
                                const U64 coordinates[] = {a, c, t, p};
                                const auto coordinate = coordinates[resolvedAxis];
                                if (coordinate != 1 && coordinate != shape[resolvedAxis] - 1) {
                                    REQUIRE(out.data<I32>()[outputIndex++] == input.at(a, c, t, p));
                                }
                            }
                        }
                    }
                }
                REQUIRE(outputIndex == out.size());
            }
        }
    }
}

TEST_CASE("Remove Indices Module - Copies every data type without conversion",
          "[modules][remove_indices][dtype]") {
    const auto implementations = Registry::ListAvailableModules("remove_indices");
    REQUIRE(!implementations.empty());

    const std::vector<DataType> types = {
        DataType::F32, DataType::F64, DataType::I8, DataType::I16, DataType::I32,
        DataType::I64, DataType::U8, DataType::U16, DataType::U32, DataType::U64,
        DataType::CF32, DataType::CF64, DataType::CI8, DataType::CI16, DataType::CI32,
        DataType::CI64, DataType::CU8, DataType::CU16, DataType::CU32, DataType::CU64,
    };
    for (const auto& impl : implementations) {
        for (const auto type : types) {
            for (const bool strided : {false, true}) {
                DYNAMIC_SECTION("Device: " << impl.device << " Type: " << type
                                << " Strided: " << strided) {
                    TestContext ctx("remove_indices", impl.device, impl.runtime, impl.provider);
                    Modules::RemoveIndices config;
                    config.indices = {0, 2, 3};
                    ctx.setConfig(config);

                    Tensor input;
                    const Shape shape = strided ? Shape{5, 3} : Shape{3, 5};
                    REQUIRE(input.create(DeviceType::CPU, type, shape) == Result::SUCCESS);
                    for (U64 i = 0; i < input.sizeBytes(); ++i) {
                        input.data<U8>()[i] = static_cast<U8>(i * 37);
                    }
                    if (strided) {
                        REQUIRE(input.permute({1, 0}) == Result::SUCCESS);
                    }
                    ctx.setInput("buffer", input);
                    REQUIRE(ctx.run() == Result::SUCCESS);

                    const auto& out = ctx.output("buffer");
                    REQUIRE(out.dtype() == type);
                    REQUIRE(out.shape() == Shape{3, 2});
                    REQUIRE(out.contiguous());
                    const U64 bytes = input.elementSize();
                    for (U64 row = 0; row < 3; ++row) {
                        const U64 kept[] = {1, 4};
                        for (U64 column = 0; column < 2; ++column) {
                            const auto* expected = input.data<U8>() +
                                input.shapeToOffset({row, kept[column]}) * bytes;
                            const auto* actual = out.data<U8>() +
                                (row * 2 + column) * bytes;
                            REQUIRE(std::memcmp(actual, expected, bytes) == 0);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Remove Indices Module - Offset and strided views",
          "[modules][remove_indices][strided]") {
    const auto implementations = Registry::ListAvailableModules("remove_indices");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const bool strided : {false, true}) {
            for (const bool remove : {false, true}) {
                DYNAMIC_SECTION("Device: " << impl.device << " Strided: " << strided
                                << " Remove: " << remove) {
                    TestContext ctx("remove_indices", impl.device, impl.runtime, impl.provider);
                    auto storage = ctx.createTensor<I32>({4, 5, 6});
                    for (U64 i = 0; i < storage.size(); ++i) {
                        storage.data()[i] = static_cast<I32>(i);
                    }
                    auto input = storage.clone();
                    if (strided) {
                        REQUIRE(input.slice({Token(1, 4), Token(0, 5), Token(1, 6, 2)}) ==
                                Result::SUCCESS);
                        REQUIRE(input.permute({1, 0, 2}) == Result::SUCCESS);
                    } else {
                        REQUIRE(input.slice({Token(1, 4), Token("...")}) == Result::SUCCESS);
                    }
                    REQUIRE(input.offset() > 0);
                    REQUIRE(input.contiguous() == !strided);

                    Modules::RemoveIndices config;
                    config.axis = strided ? 1 : 0;
                    if (remove) {
                        config.indices = {1};
                    }
                    ctx.setConfig(config);
                    ctx.setInput("buffer", input);
                    REQUIRE(ctx.run() == Result::SUCCESS);

                    const auto& out = ctx.output("buffer");
                    const U64 antennas = remove ? 2 : 3;
                    const Shape expected = strided ? Shape{5, antennas, 3} : Shape{antennas, 5, 6};
                    REQUIRE(out.shape() == expected);
                    REQUIRE(out.contiguous());
                    for (U64 a = 0; a < antennas; ++a) {
                        const U64 sourceAntenna = 1 + a + (remove && a > 0);
                        for (U64 c = 0; c < 5; ++c) {
                            for (U64 t = 0; t < (strided ? 3U : 6U); ++t) {
                                if (strided) {
                                    REQUIRE(out.at<I32>(c, a, t) == storage.at(sourceAntenna, c, 1 + 2 * t));
                                } else {
                                    REQUIRE(out.at<I32>(a, c, t) == storage.at(sourceAntenna, c, t));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Remove Indices Module - Broadcast input",
          "[modules][remove_indices][strided]") {
    const auto implementations = Registry::ListAvailableModules("remove_indices");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device) {
            TestContext ctx("remove_indices", impl.device, impl.runtime, impl.provider);
            auto input = ctx.createTensor<I32>({1, 3});
            input.at(0, 0) = 10;
            input.at(0, 1) = 20;
            input.at(0, 2) = 30;
            REQUIRE(input.broadcastTo({4, 3}) == Result::SUCCESS);
            Modules::RemoveIndices config;
            config.axis = 0;
            config.indices = {0, 2};
            ctx.setConfig(config);
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("buffer");
            REQUIRE(out.shape() == Shape{2, 3});
            REQUIRE(out.contiguous());
            for (U64 a = 0; a < 2; ++a) {
                REQUIRE(out.at<I32>(a, 0) == 10);
                REQUIRE(out.at<I32>(a, 1) == 20);
                REQUIRE(out.at<I32>(a, 2) == 30);
            }
        }
    }
}

TEST_CASE("Remove Indices Module - Validation rejects invalid selections",
          "[modules][remove_indices][validation]") {
    const auto implementations = Registry::ListAvailableModules("remove_indices");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const auto axis : {I64{2}, I64{-3}, std::numeric_limits<I64>::min(), I64{1}}) {
            for (const auto& indices : std::vector<std::vector<U64>>{
                     {3}, {std::numeric_limits<U64>::max()}, {2, 0, 1, 2}}) {
                DYNAMIC_SECTION("Device: " << impl.device << " Axis: " << axis
                                << " First index: " << indices.front()) {
                    TestContext ctx("remove_indices", impl.device, impl.runtime, impl.provider);
                    Modules::RemoveIndices config;
                    config.axis = axis;
                    config.indices = indices;
                    ctx.setConfig(config);
                    auto input = ctx.createTensor<F32>({2, 3});
                    ctx.setInput("buffer", input);
                    REQUIRE(ctx.start() == Result::ERROR);
                }
            }
        }
    }
}

TEST_CASE("Remove Indices Module - Reconfiguration leaves the live selection intact",
          "[modules][remove_indices][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("remove_indices");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const bool strided : {false, true}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Strided: " << strided) {
                TestContext ctx("remove_indices", impl.device, impl.runtime, impl.provider);
                Modules::RemoveIndices config;
                config.axis = 1;
                config.indices = {1};
                ctx.setConfig(config);

                auto input = ctx.createTensor<I32>({2, 3});
                for (U64 i = 0; i < input.size(); ++i) {
                    input.data()[i] = static_cast<I32>(i);
                }
                if (strided) {
                    REQUIRE(input.permute({1, 0}) == Result::SUCCESS);
                }
                ctx.setInput("buffer", input);
                REQUIRE(ctx.start() == Result::SUCCESS);
                REQUIRE(ctx.compute() == Result::SUCCESS);

                const auto outputId = ctx.output("buffer").id();
                config.axis = 0;
                config.indices = {0};
                REQUIRE(ctx.reconfigure(config, true) == Result::SUCCESS);
                REQUIRE(ctx.reconfigure(config) == Result::RECREATE);
                config.indices = {999};
                REQUIRE(ctx.reconfigure(config) == Result::ERROR);

                for (U64 i = 0; i < input.size(); ++i) {
                    input.data()[i] += 100;
                }
                REQUIRE(ctx.compute() == Result::SUCCESS);
                const auto& out = ctx.output("buffer");
                // CUDA outputs are copied into a fresh CPU snapshot by TestContext.
                if (impl.device == DeviceType::CPU) {
                    REQUIRE(out.id() == outputId);
                }
                REQUIRE(out.shape(0) == input.shape(0));
                REQUIRE(out.shape(1) == input.shape(1) - 1);
                for (U64 a = 0; a < out.shape(0); ++a) {
                    REQUIRE(out.at<I32>(a, 0) == input.at(a, 0));
                    if (out.shape(1) == 2) {
                        REQUIRE(out.at<I32>(a, 1) == input.at(a, 2));
                    }
                }
                REQUIRE(ctx.stop() == Result::SUCCESS);
            }
        }
    }
}
