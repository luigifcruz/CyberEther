#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/fft/module.hh"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

using namespace Jetstream;

namespace {

Tensor MakeFftTensor(const Registry::ModuleRegistration& impl,
                     const DataType dtype,
                     const Shape& shape,
                     const bool broadcast = false) {
    Tensor input;
    if (shape.empty()) {
        REQUIRE(input.create(impl.device, dtype, {1}) == Result::SUCCESS);
        REQUIRE(input.squeezeDims(0) == Result::SUCCESS);
    } else if (broadcast) {
        REQUIRE(input.create(impl.device, dtype, Shape(shape.size(), 1)) ==
                Result::SUCCESS);
        REQUIRE(input.broadcastTo(shape) == Result::SUCCESS);
    } else {
        REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    }
    return input;
}

void RequireFftValidationError(const Registry::ModuleRegistration& impl,
                               const Modules::Fft& config,
                               Tensor input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = std::move(input);

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("fft", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
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

TEST_CASE("FFT - Complex Real Signal F32", "[modules][fft][real][complex]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft config;
            config.complexOutput = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            input.at(0) = 1.0f;
            input.at(1) = 2.0f;
            input.at(2) = 3.0f;
            input.at(3) = 4.0f;
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE(out.shape() == Shape{3});

            const CF32 expected[] = {
                {10.0f, 0.0f},
                {-2.0f, 2.0f},
                {-2.0f, 0.0f},
            };
            for (U64 index = 0; index < out.size(); ++index) {
                REQUIRE_THAT(out.at<CF32>(index).real(),
                             Catch::Matchers::WithinAbs(expected[index].real(), 1e-4f));
                REQUIRE_THAT(out.at<CF32>(index).imag(),
                             Catch::Matchers::WithinAbs(expected[index].imag(), 1e-4f));
            }
        }
    }
}

TEST_CASE("FFT - Complex Real Signal Trailing Batch F32",
          "[modules][fft][real][complex][batch][metadata]") {
    auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);

            Modules::Fft config;
            config.complexOutput = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 2});
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            for (U64 row = 0; row < input.shape(0); ++row) {
                input.at(row, 0) = static_cast<F32>(row + 1);
                input.at(row, 1) = static_cast<F32>((row + 1) * 2);
            }
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE(out.shape() == Shape{3, 2});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});

            const CF32 expected[] = {
                {10.0f, 0.0f},
                {-2.0f, 2.0f},
                {-2.0f, 0.0f},
            };
            for (U64 frequency = 0; frequency < out.shape(0); ++frequency) {
                for (U64 column = 0; column < out.shape(1); ++column) {
                    const CF32 value = expected[frequency] * static_cast<F32>(column + 1);
                    REQUIRE_THAT(out.at<CF32>(frequency, column).real(),
                                 Catch::Matchers::WithinAbs(value.real(), 1e-4f));
                    REQUIRE_THAT(out.at<CF32>(frequency, column).imag(),
                                 Catch::Matchers::WithinAbs(value.imag(), 1e-4f));
                }
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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

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
            REQUIRE(singleton.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            singleton.at<F32>(0) = 3.0f;

            TestContext singletonCtx("fft", impl.device, impl.runtime, impl.provider);
            singletonCtx.setConfig(config);
            singletonCtx.setInput("signal", singleton);
            REQUIRE(singletonCtx.run() == Result::SUCCESS);
            REQUIRE_THAT(singletonCtx.output("signal").at<F32>(0),
                         Catch::Matchers::WithinAbs(3.0f, 1e-4f));

            Tensor odd;
            REQUIRE(odd.create(DeviceType::CPU, DataType::F32, {5}) == Result::SUCCESS);
            REQUIRE(odd.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
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
            REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);

            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            config.forward = false;
            ctx.setConfig(config);
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{0});
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

TEST_CASE("FFT - Multi-Head CF32", "[modules][fft][heads]") {
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
            REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);
            for (U64 head = 0; head < input.shape(0); ++head) {
                for (U64 i = 0; i < input.shape(1); ++i) {
                    input.at<CF32>(head, i) = CF32(static_cast<F32>(head + 1), 0.0f);
                }
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == Shape{2, 4});
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{1});
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{0});
            REQUIRE_FALSE(out.hasAttribute("batchAxis"));
            for (U64 head = 0; head < out.shape(0); ++head) {
                REQUIRE_THAT(out.at<CF32>(head, 0).real(),
                             Catch::Matchers::WithinAbs(4.0f * static_cast<F32>(head + 1), 1e-4f));
                REQUIRE_THAT(out.at<CF32>(head, 0).imag(),
                             Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                for (U64 i = 1; i < out.shape(1); ++i) {
                    REQUIRE_THAT(std::abs(out.at<CF32>(head, i)),
                                 Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                }
            }
        }
    }
}

TEST_CASE("FFT - Rank 3 Batched Heads CF32",
          "[modules][fft][batch][heads][metadata]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2, 3, 4});
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            for (U64 batch = 0; batch < input.shape(0); ++batch) {
                for (U64 head = 0; head < input.shape(1); ++head) {
                    const F32 value = static_cast<F32>((batch + 1) * (head + 1));
                    for (U64 sample = 0; sample < input.shape(2); ++sample) {
                        input.at(batch, head, sample) = CF32(value, 0.0f);
                    }
                }
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == Shape{2, 3, 4});
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{2});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{1});
            for (U64 batch = 0; batch < out.shape(0); ++batch) {
                for (U64 head = 0; head < out.shape(1); ++head) {
                    const F32 expected =
                        4.0f * static_cast<F32>((batch + 1) * (head + 1));
                    REQUIRE_THAT(out.at<CF32>(batch, head, 0).real(),
                                 Catch::Matchers::WithinAbs(expected, 1e-4f));
                    for (U64 sample = 1; sample < out.shape(2); ++sample) {
                        REQUIRE_THAT(std::abs(out.at<CF32>(batch, head, sample)),
                                     Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                    }
                }
            }
        }
    }
}

TEST_CASE("FFT - Preserves Signal Metadata With Opaque Dimensions",
          "[modules][fft][batch][channel][opaque][metadata]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2, 3, 4, 2});
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("layout", std::string("opaque-plane")) ==
                    Result::SUCCESS);
            for (U64 batch = 0; batch < input.shape(0); ++batch) {
                for (U64 channel = 0; channel < input.shape(1); ++channel) {
                    for (U64 plane = 0; plane < input.shape(3); ++plane) {
                        const F32 value = static_cast<F32>(
                            1 + batch * 6 + channel * 2 + plane);
                        for (U64 sample = 0; sample < input.shape(2); ++sample) {
                            input.at(batch, channel, sample, plane) = CF32(value, 0.0f);
                        }
                    }
                }
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& output = ctx.output("signal");
            REQUIRE(output.shape() == Shape{2, 3, 4, 2});
            REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{2});
            REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});
            REQUIRE(std::any_cast<std::string>(output.attribute("layout")) ==
                    "opaque-plane");
            for (U64 batch = 0; batch < output.shape(0); ++batch) {
                for (U64 channel = 0; channel < output.shape(1); ++channel) {
                    for (U64 plane = 0; plane < output.shape(3); ++plane) {
                        const F32 value = static_cast<F32>(
                            1 + batch * 6 + channel * 2 + plane);
                        REQUIRE_THAT(output.at<CF32>(batch, channel, 0, plane).real(),
                                     Catch::Matchers::WithinAbs(4.0f * value, 1e-4f));
                        for (U64 sample = 1; sample < output.shape(2); ++sample) {
                            REQUIRE_THAT(std::abs(output.at<CF32>(
                                             batch, channel, sample, plane)),
                                         Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("FFT - Multi-Head Strided Offset CF32",
          "[modules][fft][heads][strided][offset]") {
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
                for (U64 head = 0; head < storage.shape(2); ++head) {
                    storage.at<CF32>(1, row, head) =
                        CF32(static_cast<F32>(head + 1), 0.0f);
                }
            }

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(1), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(input.permute({1, 0}) == Result::SUCCESS);
            REQUIRE(input.shape() == Shape{3, 4});
            REQUIRE(input.offset() != 0);
            REQUIRE_FALSE(input.contiguous());
            REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == Shape{3, 4});
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{1});
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{0});
            REQUIRE_FALSE(out.hasAttribute("batchAxis"));
            for (U64 head = 0; head < out.shape(0); ++head) {
                REQUIRE_THAT(out.at<CF32>(head, 0).real(),
                             Catch::Matchers::WithinAbs(4.0f * static_cast<F32>(head + 1), 1e-4f));
                for (U64 i = 1; i < out.shape(1); ++i) {
                    REQUIRE_THAT(std::abs(out.at<CF32>(head, i)),
                                 Catch::Matchers::WithinAbs(0.0f, 1e-4f));
                }
            }
        }
    }
}

TEST_CASE("FFT - Trailing Batch Strided CF32",
          "[modules][fft][batch][metadata][strided][CF32]") {
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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);

            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            ctx.setConfig(config);
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == input.shape());
            REQUIRE(out.dtype() == input.dtype());
            REQUIRE(out.hasAttribute("source"));
            REQUIRE(std::any_cast<std::string>(out.attribute("source")) == "strided-view");
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{0});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});

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

TEST_CASE("FFT - Trailing Batch Invert CF32",
          "[modules][fft][batch][metadata][invert][CF32]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            config.invert = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({4, 2});
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
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
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});
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

TEST_CASE("FFT - Trailing Batch Invert FFTPACK F32",
          "[modules][fft][batch][metadata][invert][real][fftpack]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fft", impl.device, impl.runtime, impl.provider);
            Modules::Fft config;
            config.invert = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 2});
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
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
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});
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

TEST_CASE("FFT - Validation rejects missing or malformed signal metadata",
          "[modules][fft][validation]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Fft config;
            RequireFftValidationError(impl, config,
                                      MakeFftTensor(impl, DataType::F32, {}));

            RequireFftValidationError(
                impl, config, MakeFftTensor(impl, DataType::CF32, {2, 4}));

            Tensor duplicateRoles = MakeFftTensor(impl, DataType::CF32, {4});
            REQUIRE(duplicateRoles.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);
            REQUIRE(duplicateRoles.setAttribute("batchAxis", Index{0}) ==
                    Result::SUCCESS);
            RequireFftValidationError(impl, config, duplicateRoles);

            Tensor wrongType = MakeFftTensor(impl, DataType::CF32, {2, 3});
            REQUIRE(wrongType.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(wrongType.setAttribute("batchAxis", I64{0}) == Result::SUCCESS);
            RequireFftValidationError(impl, config, wrongType);

            Tensor outOfRange = MakeFftTensor(impl, DataType::CF32, {2, 3});
            REQUIRE(outOfRange.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(outOfRange.setAttribute("batchAxis", Index{2}) ==
                    Result::SUCCESS);
            RequireFftValidationError(impl, config, outOfRange);

            Tensor malformedSample = MakeFftTensor(impl, DataType::CF32, {2, 3});
            REQUIRE(malformedSample.setAttribute("sampleAxis", I64{1}) ==
                    Result::SUCCESS);
            RequireFftValidationError(impl, config, malformedSample);

            Tensor duplicateChannel = MakeFftTensor(impl, DataType::CF32, {2, 3});
            REQUIRE(duplicateChannel.setAttribute("sampleAxis", Index{1}) ==
                    Result::SUCCESS);
            REQUIRE(duplicateChannel.setAttribute("channelAxis", Index{1}) ==
                    Result::SUCCESS);
            RequireFftValidationError(impl, config, duplicateChannel);

            Tensor unsupported = MakeFftTensor(impl, DataType::F64, {4});
            REQUIRE(unsupported.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);
            RequireFftValidationError(impl, config, unsupported);
        }
    }
}

TEST_CASE("FFT - CPU validation rejects static pocketfft bounds",
          "[modules][fft][validation][cpu][bounds]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const U64 extent =
                static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) /
                    sizeof(F32) +
                1;
            Tensor input = MakeFftTensor(impl, DataType::F32, {extent}, true);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            RequireFftValidationError(impl, Modules::Fft{}, input);
        }
    }
}

TEST_CASE("FFT - CUDA validation rejects static layout grid bounds",
          "[modules][fft][validation][cuda][bounds]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    bool cudaAvailable = false;
    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CUDA) {
            continue;
        }
        cudaAvailable = true;

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const U64 extent =
                static_cast<U64>(std::numeric_limits<I32>::max()) * 256 + 1;
            Tensor input = MakeFftTensor(impl, DataType::F32, {extent}, true);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            RequireFftValidationError(impl, Modules::Fft{}, input);
        }
    }

    if (!cudaAvailable) {
        SUCCEED("CUDA FFT module is unavailable in this build.");
    }
}

TEST_CASE("FFT - Trailing Batch Roundtrip CF32",
          "[modules][fft][batch][metadata][inverse][roundtrip][CF32]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext forwardCtx("fft", impl.device, impl.runtime, impl.provider);
            auto input = forwardCtx.createTensor<CF32>({3, 2});
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            for (U64 row = 0; row < input.shape(0); ++row) {
                for (U64 column = 0; column < input.shape(1); ++column) {
                    input.at(row, column) = CF32(static_cast<F32>((row * 2) + column + 1),
                                                  static_cast<F32>(row) - 1.0f);
                }
            }

            Modules::Fft forwardConfig;
            forwardCtx.setConfig(forwardConfig);
            forwardCtx.setInput("signal", input);
            REQUIRE(forwardCtx.run() == Result::SUCCESS);
            REQUIRE(forwardCtx.output("signal").hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(
                        forwardCtx.output("signal").attribute("batchAxis")) == Index{1});

            Modules::Fft inverseConfig;
            inverseConfig.forward = false;
            TestContext inverseCtx("fft", impl.device, impl.runtime, impl.provider);
            inverseCtx.setConfig(inverseConfig);
            inverseCtx.setInput("signal", forwardCtx.output("signal"));
            REQUIRE(inverseCtx.run() == Result::SUCCESS);

            const auto& out = inverseCtx.output("signal");
            REQUIRE(out.shape() == input.shape());
            REQUIRE(out.dtype() == input.dtype());
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});
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

TEST_CASE("FFT - Trailing Batch Roundtrip FFTPACK F32",
          "[modules][fft][batch][metadata][inverse][roundtrip][real][fftpack]") {
    const auto implementations = Registry::ListAvailableModules("fft");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext forwardCtx("fft", impl.device, impl.runtime, impl.provider);
            auto input = forwardCtx.createTensor<F32>({5, 2});
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            for (U64 row = 0; row < input.shape(0); ++row) {
                for (U64 column = 0; column < input.shape(1); ++column) {
                    input.at(row, column) = static_cast<F32>((row * 2) + column + 1);
                }
            }

            Modules::Fft forwardConfig;
            forwardCtx.setConfig(forwardConfig);
            forwardCtx.setInput("signal", input);
            REQUIRE(forwardCtx.run() == Result::SUCCESS);

            Modules::Fft inverseConfig;
            inverseConfig.forward = false;
            TestContext inverseCtx("fft", impl.device, impl.runtime, impl.provider);
            inverseCtx.setConfig(inverseConfig);
            inverseCtx.setInput("signal", forwardCtx.output("signal"));
            REQUIRE(inverseCtx.run() == Result::SUCCESS);

            const auto& out = inverseCtx.output("signal");
            REQUIRE(out.shape() == input.shape());
            REQUIRE(out.dtype() == input.dtype());
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});
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
            REQUIRE(complexInput.setAttribute("sampleAxis", Index{1}) ==
                    Result::SUCCESS);
            REQUIRE(complexInput.setAttribute("batchAxis", Index{0}) ==
                    Result::SUCCESS);
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
            REQUIRE(realInput.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
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
