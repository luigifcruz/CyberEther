#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/fft/module.hh"

#include <cmath>

using namespace Jetstream;

TEST_CASE("FFT - DC Signal CF32", "[modules][fft][dc]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft config;
            config.forward = true;

            ctx.setConfig(config);

            // Create a DC signal (constant value).
            const U64 bufferSize = 64;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

            const F32 dcValue = 1.0f;
            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(dcValue, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // DC signal should produce a spike at bin 0.
            const F32 expectedDcBin = dcValue * static_cast<F32>(bufferSize);
            REQUIRE_THAT(std::abs(out.at<CF32>(0).real()), Catch::Matchers::WithinAbs(expectedDcBin, 1e-3f));
            REQUIRE_THAT(std::abs(out.at<CF32>(0).imag()), Catch::Matchers::WithinAbs(0.0f, 1e-3f));

            // All other bins should be near zero.
            for (U64 i = 1; i < bufferSize; ++i) {
                REQUIRE_THAT(std::abs(out.at<CF32>(i).real()), Catch::Matchers::WithinAbs(0.0f, 1e-3f));
                REQUIRE_THAT(std::abs(out.at<CF32>(i).imag()), Catch::Matchers::WithinAbs(0.0f, 1e-3f));
            }
        }
    }
}

TEST_CASE("FFT - Forward/Inverse Roundtrip CF32", "[modules][fft][roundtrip]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            // Forward FFT.
            TestContext forwardCtx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft forwardConfig;
            forwardConfig.forward = true;

            forwardCtx.setConfig(forwardConfig);

            // Create a test signal.
            const U64 bufferSize = 64;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                const F64 t = static_cast<F64>(i) / static_cast<F64>(bufferSize);
                input.at<CF32>(i) = CF32(static_cast<F32>(std::cos(2.0 * JST_PI * 4.0 * t)),
                                         static_cast<F32>(std::sin(2.0 * JST_PI * 4.0 * t)));
            }

            forwardCtx.setInput("signal", input);
            REQUIRE(forwardCtx.run() == Result::SUCCESS);

            // Inverse FFT.
            TestContext inverseCtx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft inverseConfig;
            inverseConfig.forward = false;

            inverseCtx.setConfig(inverseConfig);
            inverseCtx.setInput("signal", forwardCtx.output("signal"));

            REQUIRE(inverseCtx.run() == Result::SUCCESS);

            auto& recovered = inverseCtx.output("signal");

            // After forward+inverse, signal should be recovered (scaled by N).
            for (U64 i = 0; i < bufferSize; ++i) {
                const F32 scale = static_cast<F32>(bufferSize);
                const F32 expectedReal = input.at<CF32>(i).real() * scale;
                const F32 expectedImag = input.at<CF32>(i).imag() * scale;
                REQUIRE_THAT(recovered.at<CF32>(i).real(), Catch::Matchers::WithinAbs(expectedReal, 1e-2f));
                REQUIRE_THAT(recovered.at<CF32>(i).imag(), Catch::Matchers::WithinAbs(expectedImag, 1e-2f));
            }
        }
    }
}

TEST_CASE("FFT - FFTPACK Real Signal F32", "[modules][fft][real][fftpack]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft config;
            config.forward = true;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);
            input.at<F32>(0) = 1.0f;
            input.at<F32>(1) = 2.0f;
            input.at<F32>(2) = 3.0f;
            input.at<F32>(3) = 4.0f;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            const F32 expected[] = {10.0f, -2.0f, 2.0f, -2.0f};

            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE(out.shape() == Shape{4});
            for (U64 i = 0; i < out.size(); ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(expected[i], 1e-4f));
            }
        }
    }
}

TEST_CASE("FFT - FFTPACK Real Inverse F32", "[modules][fft][real][fftpack][inverse]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft config;
            config.forward = false;
            ctx.setConfig(config);

            Tensor storage;
            REQUIRE(storage.create(DeviceType::CPU, DataType::F32, {2, 4}) == Result::SUCCESS);
            storage.at<F32>(1, 0) = 10.0f;
            storage.at<F32>(1, 1) = -2.0f;
            storage.at<F32>(1, 2) = 2.0f;
            storage.at<F32>(1, 3) = -2.0f;

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(1), Token()}) == Result::SUCCESS);
            REQUIRE(input.contiguous());
            REQUIRE(input.offset() != 0);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            const F32 expected[] = {4.0f, 8.0f, 12.0f, 16.0f};
            for (U64 i = 0; i < out.size(); ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(expected[i], 1e-4f));
            }
        }
    }
}

TEST_CASE("FFT - FFTPACK Real Inverse Edge Lengths F32",
          "[modules][fft][real][fftpack][inverse][edge]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Fft config;
            config.forward = false;

            Tensor singleton;
            REQUIRE(singleton.create(DeviceType::CPU, DataType::F32, {1}) == Result::SUCCESS);
            singleton.at<F32>(0) = 3.0f;

            TestContext singletonCtx("fft", impl.device, impl.runtime, impl.provider);
            singletonCtx.setConfig(config);
            singletonCtx.setInput("signal", singleton);
            REQUIRE(singletonCtx.run() == Result::SUCCESS);
            REQUIRE_THAT(singletonCtx.output("signal").at<F32>(0),
                         Catch::Matchers::WithinAbs(3.0f, 1e-4f));

            Tensor odd;
            REQUIRE(odd.create(DeviceType::CPU, DataType::F32, {5}) == Result::SUCCESS);
            odd.at<F32>(0) = 0.0f;
            odd.at<F32>(1) = 0.0f;
            odd.at<F32>(2) = 0.0f;
            odd.at<F32>(3) = 0.0f;
            odd.at<F32>(4) = 1.0f;

            TestContext oddCtx("fft", impl.device, impl.runtime, impl.provider);
            oddCtx.setConfig(config);
            oddCtx.setInput("signal", odd);
            REQUIRE(oddCtx.run() == Result::SUCCESS);

            const auto& output = oddCtx.output("signal");
            for (U64 i = 0; i < odd.size(); ++i) {
                const F32 expected = static_cast<F32>(
                    -2.0 * std::sin(4.0 * JST_PI * static_cast<F64>(i) / 5.0));
                REQUIRE_THAT(output.at<F32>(i),
                             Catch::Matchers::WithinAbs(expected, 1e-4f));
            }
        }
    }
}

TEST_CASE("FFT - FFTPACK Real Odd Roundtrip F32", "[modules][fft][real][fftpack][roundtrip]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            constexpr U64 bufferSize = 7;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {bufferSize}) == Result::SUCCESS);
            const F32 samples[] = {1.0f, -2.0f, 3.5f, 0.25f, -1.25f, 2.0f, 0.5f};
            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<F32>(i) = samples[i];
            }

            TestContext forwardCtx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft forwardConfig;
            forwardConfig.forward = true;
            forwardCtx.setConfig(forwardConfig);
            forwardCtx.setInput("signal", input);
            REQUIRE(forwardCtx.run() == Result::SUCCESS);

            TestContext inverseCtx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft inverseConfig;
            inverseConfig.forward = false;
            inverseCtx.setConfig(inverseConfig);
            inverseCtx.setInput("signal", forwardCtx.output("signal"));
            REQUIRE(inverseCtx.run() == Result::SUCCESS);

            const auto& recovered = inverseCtx.output("signal");
            for (U64 i = 0; i < bufferSize; ++i) {
                REQUIRE_THAT(recovered.at<F32>(i),
                             Catch::Matchers::WithinAbs(samples[i] * bufferSize, 1e-3f));
            }
        }
    }
}

TEST_CASE("FFT - FFTPACK Real Inverse Batched Strided F32",
          "[modules][fft][real][fftpack][inverse][batch][strided]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor storage;
            REQUIRE(storage.create(DeviceType::CPU, DataType::F32, {3, 4, 2}) == Result::SUCCESS);

            const F32 packed[][4] = {
                {10.0f, -2.0f, 2.0f, -2.0f},
                {20.0f, -4.0f, 4.0f, -4.0f},
            };
            for (U64 row = 0; row < 4; ++row) {
                for (U64 batch = 0; batch < 2; ++batch) {
                    storage.at<F32>(1, row, batch) = packed[batch][row];
                }
            }

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(1), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(input.permute({1, 0}) == Result::SUCCESS);
            REQUIRE(input.shape() == Shape{2, 4});
            REQUIRE(input.offset() != 0);
            REQUIRE_FALSE(input.contiguous());

            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            config.forward = false;
            ctx.setConfig(config);
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            for (U64 batch = 0; batch < 2; ++batch) {
                for (U64 i = 0; i < 4; ++i) {
                    const F32 expected = static_cast<F32>((batch + 1) * (i + 1) * 4);
                    REQUIRE_THAT(out.at<F32>(batch, i),
                                 Catch::Matchers::WithinAbs(expected, 1e-4f));
                }
            }
        }
    }
}

TEST_CASE("FFT - Batched CF32", "[modules][fft][batch]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft config;
            config.forward = true;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {2, 4}) == Result::SUCCESS);
            for (U64 batch = 0; batch < input.shape(0); ++batch) {
                for (U64 i = 0; i < input.shape(1); ++i) {
                    input.at<CF32>(batch, i) = CF32(static_cast<F32>(batch + 1), 0.0f);
                }
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == Shape{2, 4});
            for (U64 batch = 0; batch < out.shape(0); ++batch) {
                REQUIRE_THAT(out.at<CF32>(batch, 0).real(),
                             Catch::Matchers::WithinAbs(4.0f * static_cast<F32>(batch + 1), 1e-4f));
                REQUIRE_THAT(out.at<CF32>(batch, 0).imag(),
                             Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                for (U64 i = 1; i < out.shape(1); ++i) {
                    REQUIRE_THAT(std::abs(out.at<CF32>(batch, i)),
                                 Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                }
            }
        }
    }
}

TEST_CASE("FFT - Strided Offset CF32", "[modules][fft][strided][offset]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft config;
            config.forward = true;
            ctx.setConfig(config);

            Tensor storage;
            REQUIRE(storage.create(DeviceType::CPU, DataType::CF32, {3, 4, 3}) == Result::SUCCESS);
            for (U64 row = 0; row < storage.shape(1); ++row) {
                for (U64 batch = 0; batch < storage.shape(2); ++batch) {
                    storage.at<CF32>(1, row, batch) =
                        CF32(static_cast<F32>(batch + 1), 0.0f);
                }
            }

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(1), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(input.permute({1, 0}) == Result::SUCCESS);
            REQUIRE(input.shape() == Shape{3, 4});
            REQUIRE(input.offset() != 0);
            REQUIRE_FALSE(input.contiguous());

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == Shape{3, 4});
            for (U64 batch = 0; batch < out.shape(0); ++batch) {
                REQUIRE_THAT(out.at<CF32>(batch, 0).real(),
                             Catch::Matchers::WithinAbs(4.0f * static_cast<F32>(batch + 1), 1e-4f));
                for (U64 i = 1; i < out.shape(1); ++i) {
                    REQUIRE_THAT(std::abs(out.at<CF32>(batch, i)),
                                 Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                }
            }
        }
    }
}

TEST_CASE("FFT - Negative Arbitrary Axis Strided CF32",
          "[modules][fft][axis][strided][CF32]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor storage;
            REQUIRE(storage.create(DeviceType::CPU, DataType::CF32, {2, 3, 2}) ==
                    Result::SUCCESS);
            const F32 values[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
            for (U64 row = 0; row < 2; ++row) {
                for (U64 column = 0; column < 3; ++column) {
                    storage.at<CF32>(row, column, 1) = CF32(values[row][column], 0.0f);
                }
            }

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(), Token(), Token(1)}) == Result::SUCCESS);
            REQUIRE(input.shape() == Shape{2, 3});
            REQUIRE(input.offset() != 0);
            REQUIRE_FALSE(input.contiguous());
            REQUIRE(input.setAttribute("source", std::string("strided-view")) ==
                    Result::SUCCESS);

            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            config.axis = -2;
            ctx.setConfig(config);
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == input.shape());
            REQUIRE(out.dtype() == input.dtype());
            REQUIRE(out.hasAttribute("source"));
            REQUIRE(std::any_cast<std::string>(out.attribute("source")) == "strided-view");

            const F32 expected[2][3] = {{5.0f, 7.0f, 9.0f}, {-3.0f, -3.0f, -3.0f}};
            for (U64 row = 0; row < out.shape(0); ++row) {
                for (U64 column = 0; column < out.shape(1); ++column) {
                    REQUIRE_THAT(out.at<CF32>(row, column).real(),
                                 Catch::Matchers::WithinAbs(expected[row][column], 1e-4f));
                    REQUIRE_THAT(out.at<CF32>(row, column).imag(),
                                 Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                }
            }
        }
    }
}

TEST_CASE("FFT - Arbitrary Axis Invert CF32", "[modules][fft][axis][invert][CF32]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            config.axis = 0;
            config.invert = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({4, 2});
            for (U64 row = 0; row < input.shape(0); ++row) {
                for (U64 column = 0; column < input.shape(1); ++column) {
                    input.at(row, column) = CF32(1.0f, 0.0f);
                }
            }
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == Shape{4, 2});
            REQUIRE(out.dtype() == DataType::CF32);
            for (U64 row = 0; row < out.shape(0); ++row) {
                const F32 expected = row == 2 ? 4.0f : 0.0f;
                for (U64 column = 0; column < out.shape(1); ++column) {
                    REQUIRE_THAT(out.at<CF32>(row, column).real(),
                                 Catch::Matchers::WithinAbs(expected, 1e-4f));
                    REQUIRE_THAT(out.at<CF32>(row, column).imag(),
                                 Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                }
            }
        }
    }
}

TEST_CASE("FFT - Arbitrary Axis Invert FFTPACK F32",
          "[modules][fft][axis][invert][real][fftpack]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            config.axis = 0;
            config.invert = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 2});
            for (U64 row = 0; row < input.shape(0); ++row) {
                for (U64 column = 0; column < input.shape(1); ++column) {
                    input.at(row, column) = 1.0f;
                }
            }
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == Shape{4, 2});
            REQUIRE(out.dtype() == DataType::F32);
            for (U64 row = 0; row < out.shape(0); ++row) {
                const F32 expected = row == 3 ? 4.0f : 0.0f;
                for (U64 column = 0; column < out.shape(1); ++column) {
                    REQUIRE_THAT(out.at<F32>(row, column),
                                 Catch::Matchers::WithinAbs(expected, 1e-4f));
                }
            }
        }
    }
}

TEST_CASE("FFT - Invalid Axis Error", "[modules][fft][axis][error]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            for (const I64 invalidAxis : {I64{2}, I64{-3}}) {
                DYNAMIC_SECTION("Axis: " << invalidAxis) {
                    TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
                    Modules::Fft config;
                    config.axis = invalidAxis;
                    ctx.setConfig(config);
                    auto input = ctx.createTensor<CF32>({2, 3});
                    ctx.setInput("signal", input);
                    REQUIRE(ctx.run() == Result::ERROR);
                }
            }
        }
    }
}

TEST_CASE("FFT - Arbitrary Axis Roundtrip CF32",
          "[modules][fft][axis][inverse][roundtrip][CF32]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext forwardCtx("fft", impl.device, impl.runtime, impl.provider);
            auto input = forwardCtx.createTensor<CF32>({3, 2});
            for (U64 row = 0; row < input.shape(0); ++row) {
                for (U64 column = 0; column < input.shape(1); ++column) {
                    input.at(row, column) = CF32(static_cast<F32>((row * 2) + column + 1),
                                                  static_cast<F32>(row) - 1.0f);
                }
            }

            Modules::Fft forwardConfig;
            forwardConfig.axis = 0;
            forwardCtx.setConfig(forwardConfig);
            forwardCtx.setInput("signal", input);
            REQUIRE(forwardCtx.run() == Result::SUCCESS);

            Modules::Fft inverseConfig;
            inverseConfig.forward = false;
            inverseConfig.axis = -2;
            TestContext inverseCtx("fft", impl.device, impl.runtime, impl.provider);
            inverseCtx.setConfig(inverseConfig);
            inverseCtx.setInput("signal", forwardCtx.output("signal"));
            REQUIRE(inverseCtx.run() == Result::SUCCESS);

            const auto& out = inverseCtx.output("signal");
            REQUIRE(out.shape() == input.shape());
            REQUIRE(out.dtype() == input.dtype());
            for (U64 row = 0; row < out.shape(0); ++row) {
                for (U64 column = 0; column < out.shape(1); ++column) {
                    REQUIRE_THAT(out.at<CF32>(row, column).real(),
                                 Catch::Matchers::WithinAbs(input.at(row, column).real() * 3.0f,
                                                            1e-4f));
                    REQUIRE_THAT(out.at<CF32>(row, column).imag(),
                                 Catch::Matchers::WithinAbs(input.at(row, column).imag() * 3.0f,
                                                            1e-4f));
                }
            }
        }
    }
}

TEST_CASE("FFT - Arbitrary Axis Roundtrip FFTPACK F32",
          "[modules][fft][axis][inverse][roundtrip][real][fftpack]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext forwardCtx("fft", impl.device, impl.runtime, impl.provider);
            auto input = forwardCtx.createTensor<F32>({5, 2});
            for (U64 row = 0; row < input.shape(0); ++row) {
                for (U64 column = 0; column < input.shape(1); ++column) {
                    input.at(row, column) = static_cast<F32>((row * 2) + column + 1);
                }
            }

            Modules::Fft forwardConfig;
            forwardConfig.axis = 0;
            forwardCtx.setConfig(forwardConfig);
            forwardCtx.setInput("signal", input);
            REQUIRE(forwardCtx.run() == Result::SUCCESS);

            Modules::Fft inverseConfig;
            inverseConfig.forward = false;
            inverseConfig.axis = 0;
            TestContext inverseCtx("fft", impl.device, impl.runtime, impl.provider);
            inverseCtx.setConfig(inverseConfig);
            inverseCtx.setInput("signal", forwardCtx.output("signal"));
            REQUIRE(inverseCtx.run() == Result::SUCCESS);

            const auto& out = inverseCtx.output("signal");
            REQUIRE(out.shape() == input.shape());
            REQUIRE(out.dtype() == input.dtype());
            for (U64 row = 0; row < out.shape(0); ++row) {
                for (U64 column = 0; column < out.shape(1); ++column) {
                    REQUIRE_THAT(out.at<F32>(row, column),
                                 Catch::Matchers::WithinAbs(input.at(row, column) * 5.0f,
                                                            1e-3f));
                }
            }
        }
    }
}

TEST_CASE("FFT - CUDA Recreation Resets Execution Path",
          "[modules][fft][cuda][recreate]") {
    const auto implementations =
        Registry::ListAvailableModules("fft", DeviceType::CUDA);
    if (implementations.empty()) {
        SUCCEED("CUDA FFT module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("fft",
                                          impl.device,
                                          impl.runtime,
                                          impl.provider,
                                          module) == Result::SUCCESS);

            TypedTensor<CF32> complexInput(DeviceType::CPU, {2, 4});
            for (U64 index = 0; index < complexInput.size(); ++index) {
                complexInput.at(index) = CF32(1.0f, 0.0f);
            }

            Tensor complexDeviceInput(impl.device, complexInput);
            TensorMap complexInputs;
            complexInputs["signal"].requested("test", "signal");
            complexInputs["signal"].tensor = complexDeviceInput;

            Modules::Fft complexConfig;
            complexConfig.invert = true;
            REQUIRE(module->create("test", complexConfig, complexInputs) == Result::SUCCESS);

            Runtime complexRuntime("test", impl.device, impl.runtime);
            REQUIRE(complexRuntime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(complexRuntime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
            REQUIRE(complexRuntime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);

            TypedTensor<F32> realInput(DeviceType::CPU, {4});
            realInput.at(0) = 1.0f;
            realInput.at(1) = 2.0f;
            realInput.at(2) = 3.0f;
            realInput.at(3) = 4.0f;

            Tensor realDeviceInput(impl.device, realInput);
            TensorMap realInputs;
            realInputs["signal"].requested("test", "signal");
            realInputs["signal"].tensor = realDeviceInput;

            Modules::Fft realConfig;
            REQUIRE(module->create("test", realConfig, realInputs) == Result::SUCCESS);

            Runtime realRuntime("test", impl.device, impl.runtime);
            REQUIRE(realRuntime.create({{"test", module}}) == Result::SUCCESS);
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(realRuntime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

            Tensor output(DeviceType::CPU, module->outputs().at("signal").tensor);
            const F32 expected[] = {10.0f, -2.0f, 2.0f, -2.0f};
            for (U64 index = 0; index < output.size(); ++index) {
                REQUIRE_THAT(output.at<F32>(index),
                             Catch::Matchers::WithinAbs(expected[index], 1e-4f));
            }

            REQUIRE(realRuntime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
