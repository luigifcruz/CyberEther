#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/amplitude/module.hh"

#include <cmath>
#include <limits>
#include <string>

using namespace Jetstream;

namespace {

Tensor MakeAmplitudeTensor(const Registry::ModuleRegistration& impl,
                           const DataType dtype,
                           const Shape& shape,
                           const bool broadcast = false) {
    Tensor input;
    if (broadcast) {
        REQUIRE(input.create(impl.device, dtype, {1}) == Result::SUCCESS);
        REQUIRE(input.broadcastTo(shape) == Result::SUCCESS);
    } else if (shape.empty()) {
        REQUIRE(input.create(impl.device, dtype, {1}) == Result::SUCCESS);
        REQUIRE(input.squeezeDims(0) == Result::SUCCESS);
    } else {
        REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    }
    return input;
}

void RequireAmplitudeValidationError(const Registry::ModuleRegistration& impl,
                                     const Modules::Amplitude& config,
                                     Tensor input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("amplitude", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Amplitude - CF32 DC Signal", "[modules][amplitude][cf32]") {
    auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Modules::Amplitude config;

            ctx.setConfig(config);

            // Create a constant complex signal.
            const U64 bufferSize = 64;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

            const F32 magnitude = 1.0f;
            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(magnitude, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // All output values should be the same (constant input).
            // Expected: 20*log10(1.0) + 20*log10(1/64) = 0 + (-36.12) ≈ -36.12 dB
            const F32 scalingCoeff = 20.0f * std::log10(1.0f / static_cast<F32>(bufferSize));
            const F32 expected = 20.0f * std::log10(magnitude) + scalingCoeff;

            for (U64 i = 0; i < bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(expected, 0.5f));
            }
        }
    }
}

TEST_CASE("Amplitude - F32 Signal", "[modules][amplitude][f32]") {
    auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Modules::Amplitude config;

            ctx.setConfig(config);

            // Create a constant real signal.
            const U64 bufferSize = 128;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {bufferSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            const F32 value = 2.0f;
            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<F32>(i) = value;
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // Expected: 20*log10(2.0) + 20*log10(1/128) ≈ 6.02 + (-42.14) ≈ -36.12 dB
            const F32 scalingCoeff = 20.0f * std::log10(1.0f / static_cast<F32>(bufferSize));
            const F32 expected = 20.0f * std::log10(value) + scalingCoeff;

            for (U64 i = 0; i < bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(expected, 0.5f));
            }
        }
    }
}

TEST_CASE("Amplitude - Channel-only Signal",
          "[modules][amplitude][channel][metadata]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);
            ctx.setConfig(Modules::Amplitude{});

            auto input = ctx.createTensor<F32>({5, 2});
            REQUIRE(input.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            for (U64 index = 0; index < input.size(); ++index) {
                input.data()[index] = 5.0f;
            }
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE_FALSE(out.hasAttribute("sampleAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});
            const F32 expected = 20.0f * std::log10(5.0f);
            for (U64 index = 0; index < out.size(); ++index) {
                REQUIRE_THAT(out.data<F32>()[index],
                             Catch::Matchers::WithinAbs(expected, 0.1f));
            }
        }
    }
}

TEST_CASE("Amplitude - Trailing Batch Metadata Normalization",
          "[modules][amplitude][batch][metadata]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Modules::Amplitude config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({5, 3});
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            for (U64 index = 0; index < input.size(); ++index) {
                input.data()[index] = 5.0f;
            }
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{0});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});
            for (U64 index = 0; index < out.size(); ++index) {
                REQUIRE_THAT(out.data<F32>()[index],
                             Catch::Matchers::WithinAbs(0.0f, 0.1f));
            }
        }
    }
}

TEST_CASE("Amplitude - Leading Batch Metadata Normalization",
          "[modules][amplitude][batch][metadata]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Modules::Amplitude config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({3, 5});
            REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            for (U64 index = 0; index < input.size(); ++index) {
                input.data()[index] = 5.0f;
            }
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& out = ctx.output("signal");
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{1});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{0});
            for (U64 index = 0; index < out.size(); ++index) {
                REQUIRE_THAT(out.data<F32>()[index],
                             Catch::Matchers::WithinAbs(0.0f, 0.1f));
            }
        }
    }
}

TEST_CASE("Amplitude - Rank 3 Batched Heads Normalization",
          "[modules][amplitude][batch][heads][metadata]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);
            Modules::Amplitude config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3, 4});
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            for (U64 batch = 0; batch < input.shape(0); ++batch) {
                for (U64 head = 0; head < input.shape(1); ++head) {
                    const F32 value =
                        4.0f * static_cast<F32>((batch + 1) * (head + 1));
                    for (U64 sample = 0; sample < input.shape(2); ++sample) {
                        input.at(batch, head, sample) = value;
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
                    const F32 expected = 20.0f * std::log10(
                        static_cast<F32>((batch + 1) * (head + 1)));
                    for (U64 sample = 0; sample < out.shape(2); ++sample) {
                        REQUIRE_THAT(out.at<F32>(batch, head, sample),
                                     Catch::Matchers::WithinAbs(expected, 0.1f));
                    }
                }
            }
        }
    }
}

TEST_CASE("Amplitude - Validation rejects missing or malformed signal metadata",
          "[modules][amplitude][validation][layout]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Amplitude config;

            RequireAmplitudeValidationError(
                impl, config, MakeAmplitudeTensor(impl, DataType::F32, {2, 3}));

            Tensor duplicateRoles = MakeAmplitudeTensor(impl, DataType::F32, {4});
            REQUIRE(duplicateRoles.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);
            REQUIRE(duplicateRoles.setAttribute("batchAxis", Index{0}) ==
                    Result::SUCCESS);
            RequireAmplitudeValidationError(impl, config, duplicateRoles);

            Tensor wrongType =
                MakeAmplitudeTensor(impl, DataType::F32, {2, 3});
            REQUIRE(wrongType.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(wrongType.setAttribute("batchAxis", I64{0}) == Result::SUCCESS);
            RequireAmplitudeValidationError(impl, config, wrongType);

            Tensor outOfRange =
                MakeAmplitudeTensor(impl, DataType::F32, {2, 3});
            REQUIRE(outOfRange.setAttribute("sampleAxis", Index{1}) ==
                    Result::SUCCESS);
            REQUIRE(outOfRange.setAttribute("batchAxis", Index{2}) ==
                    Result::SUCCESS);
            RequireAmplitudeValidationError(impl, config, outOfRange);

            Tensor malformedSample =
                MakeAmplitudeTensor(impl, DataType::F32, {2, 3});
            REQUIRE(malformedSample.setAttribute("sampleAxis", I64{1}) ==
                    Result::SUCCESS);
            RequireAmplitudeValidationError(impl, config, malformedSample);

            Tensor duplicateChannel =
                MakeAmplitudeTensor(impl, DataType::F32, {2, 3});
            REQUIRE(duplicateChannel.setAttribute("sampleAxis", Index{1}) ==
                    Result::SUCCESS);
            REQUIRE(duplicateChannel.setAttribute("channelAxis", Index{1}) ==
                    Result::SUCCESS);
            RequireAmplitudeValidationError(impl, config, duplicateChannel);
        }
    }
}

TEST_CASE("Amplitude - Validation rejects rank zero",
          "[modules][amplitude][validation][rank]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Amplitude config;
            RequireAmplitudeValidationError(
                impl, config, MakeAmplitudeTensor(impl, DataType::F32, {}));
        }
    }
}

TEST_CASE("Amplitude - Validation rejects unsupported dtype",
          "[modules][amplitude][validation][dtype]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Amplitude config;
            Tensor input = MakeAmplitudeTensor(impl, DataType::F64, {4});
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            RequireAmplitudeValidationError(impl, config, input);
        }
    }
}

TEST_CASE("Amplitude - CUDA validation rejects unsupported grid size",
          "[modules][amplitude][validation][cuda]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CUDA) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Amplitude config;
            const Shape shape = {
                static_cast<U64>(std::numeric_limits<I32>::max()) * 256 + 1,
            };
            Tensor input = MakeAmplitudeTensor(impl, DataType::F32, shape, true);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            RequireAmplitudeValidationError(impl, config, input);
        }
    }
}

TEST_CASE("Amplitude - CF32 Various Magnitudes", "[modules][amplitude][magnitude]") {
    auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Modules::Amplitude config;

            ctx.setConfig(config);

            const U64 bufferSize = 4;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            // Create complex samples with known magnitudes: 1, 2, 3, 4
            input.at<CF32>(0) = CF32(1.0f, 0.0f);   // magnitude = 1
            input.at<CF32>(1) = CF32(0.0f, 2.0f);   // magnitude = 2
            input.at<CF32>(2) = CF32(2.4f, 1.8f);   // magnitude = 3 (3-4-5 triangle scaled)
            input.at<CF32>(3) = CF32(4.0f, 0.0f);   // magnitude = 4

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            const F32 scalingCoeff = 20.0f * std::log10(1.0f / static_cast<F32>(bufferSize));

            // Verify each output corresponds to 20*log10(magnitude) + scalingCoeff
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(20.0f * std::log10(1.0f) + scalingCoeff, 0.5f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(20.0f * std::log10(2.0f) + scalingCoeff, 0.5f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(20.0f * std::log10(3.0f) + scalingCoeff, 0.5f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(20.0f * std::log10(4.0f) + scalingCoeff, 0.5f));
        }
    }
}

TEST_CASE("Amplitude - F32 exact zero is negative infinity",
          "[modules][amplitude][f32][zero]") {
    auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);
            Modules::Amplitude config;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            input.at<F32>(0) = 0.0f;
            input.at<F32>(1) = 1.0f;
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            REQUIRE(std::isinf(out.at<F32>(0)));
            REQUIRE(std::signbit(out.at<F32>(0)));
            REQUIRE(std::isfinite(out.at<F32>(1)));
        }
    }
}

TEST_CASE("Amplitude - CF32 exact zero is negative infinity",
          "[modules][amplitude][cf32][zero]") {
    auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);
            Modules::Amplitude config;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            input.at<CF32>(0) = CF32(0.0f, 0.0f);
            input.at<CF32>(1) = CF32(0.0f, 1.0f);
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            REQUIRE(std::isinf(out.at<F32>(0)));
            REQUIRE(std::signbit(out.at<F32>(0)));
            REQUIRE(std::isfinite(out.at<F32>(1)));
        }
    }
}

TEST_CASE("Amplitude - Rank 4 Non-Contiguous F32 With Opaque Dimension",
          "[modules][amplitude][f32][noncontiguous][metadata]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Tensor storage(DeviceType::CPU, DataType::F32, {2, 2, 3, 2, 4});
            for (U64 i = 0; i < storage.size(); ++i) {
                storage.data<F32>()[i] = 0.5f + static_cast<F32>(i) * 0.125f;
            }

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(1), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(input.permute({1, 0, 3, 2}) == Result::SUCCESS);
            REQUIRE(input.shape() == Shape{3, 2, 4, 2});
            REQUIRE(input.offset() != 0);
            REQUIRE_FALSE(input.contiguous());
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("layout", std::string("opaque-plane")) ==
                    Result::SUCCESS);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{2});
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{1});
            REQUIRE(std::any_cast<std::string>(out.attribute("layout")) ==
                    "opaque-plane");
            const F32 scalingCoeff = 20.0f * std::log10(0.25f);
            for (U64 i = 0; i < 3; ++i) {
                for (U64 j = 0; j < 2; ++j) {
                    for (U64 k = 0; k < 4; ++k) {
                        for (U64 l = 0; l < 2; ++l) {
                            const F32 expected = 20.0f * std::log10(std::fabs(
                                input.at<F32>(i, j, k, l))) + scalingCoeff;
                            REQUIRE_THAT(out.at<F32>(i, j, k, l),
                                         Catch::Matchers::WithinAbs(expected, 0.1f));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Amplitude - Rank 4 Non-Contiguous CF32 With Opaque Dimension",
          "[modules][amplitude][cf32][noncontiguous][metadata]") {
    const auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Tensor storage(DeviceType::CPU, DataType::CF32, {2, 2, 3, 2, 4});
            for (U64 i = 0; i < storage.size(); ++i) {
                const F32 value = 0.5f + static_cast<F32>(i) * 0.125f;
                storage.data<CF32>()[i] = CF32(value, value * 0.5f);
            }

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(1), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(input.permute({1, 0, 3, 2}) == Result::SUCCESS);
            REQUIRE(input.shape() == Shape{3, 2, 4, 2});
            REQUIRE(input.offset() != 0);
            REQUIRE_FALSE(input.contiguous());
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("layout", std::string("opaque-plane")) ==
                    Result::SUCCESS);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{2});
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{1});
            REQUIRE(std::any_cast<std::string>(out.attribute("layout")) ==
                    "opaque-plane");
            const F32 scalingCoeff = 20.0f * std::log10(0.25f);
            for (U64 i = 0; i < 3; ++i) {
                for (U64 j = 0; j < 2; ++j) {
                    for (U64 k = 0; k < 4; ++k) {
                        for (U64 l = 0; l < 2; ++l) {
                            const CF32 value = input.at<CF32>(i, j, k, l);
                            const F32 magnitude = std::sqrt(
                                value.real() * value.real() + value.imag() * value.imag());
                            const F32 expected = 20.0f * std::log10(magnitude) + scalingCoeff;
                            REQUIRE_THAT(out.at<F32>(i, j, k, l),
                                         Catch::Matchers::WithinAbs(expected, 0.1f));
                        }
                    }
                }
            }
        }
    }
}
