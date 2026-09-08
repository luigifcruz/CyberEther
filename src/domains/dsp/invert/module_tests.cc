#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <any>
#include <cmath>
#include <string>

#include "jetstream/registry.hh"
#include "jetstream/testing.hh"
#include "jetstream/domains/dsp/invert/module.hh"

using namespace Jetstream;

namespace {

void RequireInvertValidationError(const Registry::ModuleRegistration& impl,
                                   const DataType dtype,
                                   const Shape& shape,
                                   const std::any& sampleAxis = {},
                                   const std::any& batchAxis = {},
                                   const std::any& channelAxis = {}) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    if (sampleAxis.has_value()) {
        REQUIRE(input.setAttribute("sampleAxis", sampleAxis) == Result::SUCCESS);
    }
    if (batchAxis.has_value()) {
        REQUIRE(input.setAttribute("batchAxis", batchAxis) == Result::SUCCESS);
    }
    if (channelAxis.has_value()) {
        REQUIRE(input.setAttribute("channelAxis", channelAxis) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("invert", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::Invert{}, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Invert Module - Even Length Alternating Sign", "[modules][invert][CF32]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<CF32>({4});
            input.at(0) = {1.0f, 1.0f};
            input.at(1) = {2.0f, -2.0f};
            input.at(2) = {3.0f, 3.0f};
            input.at(3) = {4.0f, -4.0f};

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::CF32);
            for (U64 sample = 0; sample < input.size(); ++sample) {
                const F32 sign = (sample & 1ULL) != 0 ? -1.0f : 1.0f;
                REQUIRE(out.at<CF32>(sample) == input.at(sample) * sign);
            }
        }
    }
}

TEST_CASE("Invert Module - Even Length F32 Promotes To CF32",
          "[modules][invert][F32][CF32]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<F32>({4});
            for (U64 sample = 0; sample < input.size(); ++sample) {
                input.at(sample) = static_cast<F32>(sample + 1);
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::CF32);
            for (U64 sample = 0; sample < input.size(); ++sample) {
                const F32 sign = (sample & 1ULL) != 0 ? -1.0f : 1.0f;
                REQUIRE(out.at<CF32>(sample) == CF32(input.at(sample) * sign, 0.0f));
            }
        }
    }
}

TEST_CASE("Invert Module - Odd Length Integer Bin Shift", "[modules][invert][CF32]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<CF32>({5});
            for (U64 sample = 0; sample < input.size(); ++sample) {
                const F32 value = static_cast<F32>(sample + 1);
                input.at(sample) = {value, -0.5f * value};
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::CF32);
            const U64 binShift = input.size() / 2;
            for (U64 sample = 0; sample < input.size(); ++sample) {
                const F64 phase = 2.0 * JST_PI *
                                  static_cast<F64>(binShift) *
                                  static_cast<F64>(sample) /
                                  static_cast<F64>(input.size());
                const CF32 phasor{static_cast<F32>(std::cos(phase)),
                                  static_cast<F32>(std::sin(phase))};
                const CF32 expected = input.at(sample) * phasor;
                REQUIRE_THAT(out.at<CF32>(sample).real(),
                             Catch::Matchers::WithinAbs(expected.real(), 1e-5f));
                REQUIRE_THAT(out.at<CF32>(sample).imag(),
                             Catch::Matchers::WithinAbs(expected.imag(), 1e-5f));
            }
        }
    }
}

TEST_CASE("Invert Module - Odd Length F32 Promotes To CF32",
          "[modules][invert][F32][CF32]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<F32>({5});
            for (U64 sample = 0; sample < input.size(); ++sample) {
                input.at(sample) = static_cast<F32>(sample + 1);
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::CF32);
            const U64 binShift = input.size() / 2;
            for (U64 sample = 0; sample < input.size(); ++sample) {
                const F64 phase = 2.0 * JST_PI *
                                  static_cast<F64>(binShift) *
                                  static_cast<F64>(sample) /
                                  static_cast<F64>(input.size());
                const CF32 expected =
                    CF32(input.at(sample), 0.0f) *
                    CF32(static_cast<F32>(std::cos(phase)),
                         static_cast<F32>(std::sin(phase)));
                REQUIRE_THAT(out.at<CF32>(sample).real(),
                             Catch::Matchers::WithinAbs(expected.real(), 1e-5f));
                REQUIRE_THAT(out.at<CF32>(sample).imag(),
                             Catch::Matchers::WithinAbs(expected.imag(), 1e-5f));
            }
        }
    }
}

TEST_CASE("Invert Module - Unsupported DType Error", "[modules][invert][error]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireInvertValidationError(impl, DataType::F64, {3}, Index{0});
        }
    }
}

TEST_CASE("Invert Module - Leading Batch Restarts For Each Batch",
          "[modules][invert][batch][leading]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<CF32>({2, 3});
            REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            for (U64 row = 0; row < input.shape(0); ++row) {
                for (U64 column = 0; column < input.shape(1); ++column) {
                    input.at(row, column) = CF32(static_cast<F32>((row * 3) + column + 1),
                                                  static_cast<F32>(column + 1));
                }
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{1});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == 0);
            const F64 phaseStep = 2.0 * JST_PI *
                                  static_cast<F64>(input.shape(1) / 2) /
                                  static_cast<F64>(input.shape(1));
            for (U64 row = 0; row < out.shape(0); ++row) {
                for (U64 column = 0; column < out.shape(1); ++column) {
                    const F64 phase = phaseStep * static_cast<F64>(column);
                    const CF32 phasor{static_cast<F32>(std::cos(phase)),
                                      static_cast<F32>(std::sin(phase))};
                    const CF32 expected = input.at(row, column) * phasor;
                    REQUIRE_THAT(out.at<CF32>(row, column).real(),
                                 Catch::Matchers::WithinAbs(expected.real(), 1e-5f));
                    REQUIRE_THAT(out.at<CF32>(row, column).imag(),
                                 Catch::Matchers::WithinAbs(expected.imag(), 1e-5f));
                }
            }
        }
    }
}

TEST_CASE("Invert Module - Multi-Head Restarts For Each Head",
          "[modules][invert][head]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<CF32>({2, 4});
            REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);
            for (U64 head = 0; head < input.shape(0); ++head) {
                for (U64 sample = 0; sample < input.shape(1); ++sample) {
                    const F32 value = static_cast<F32>((head * 4) + sample + 1);
                    input.at(head, sample) = CF32(value, -value);
                }
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{1});
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{0});
            REQUIRE_FALSE(out.hasAttribute("batchAxis"));
            for (U64 head = 0; head < out.shape(0); ++head) {
                for (U64 sample = 0; sample < out.shape(1); ++sample) {
                    const F32 sign = (sample & 1ULL) != 0 ? -1.0f : 1.0f;
                    REQUIRE(out.at<CF32>(head, sample) == input.at(head, sample) * sign);
                }
            }
        }
    }
}

TEST_CASE("Invert Module - Rank-4 Batched Multi-Head With Opaque Planes",
          "[modules][invert][batch][head][opaque]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<CF32>({2, 3, 4, 2});
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("layout", std::string("opaque-plane")) ==
                    Result::SUCCESS);
            for (U64 batch = 0; batch < input.shape(0); ++batch) {
                for (U64 head = 0; head < input.shape(1); ++head) {
                    for (U64 sample = 0; sample < input.shape(2); ++sample) {
                        for (U64 plane = 0; plane < input.shape(3); ++plane) {
                            const F32 value = static_cast<F32>(
                                (batch * 24) + (head * 8) + (sample * 2) + plane + 1);
                            input.at(batch, head, sample, plane) =
                                CF32(value, value * 0.5f);
                        }
                    }
                }
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{2});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == 0);
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{1});
            REQUIRE(std::any_cast<std::string>(out.attribute("layout")) ==
                    "opaque-plane");
            for (U64 batch = 0; batch < out.shape(0); ++batch) {
                for (U64 head = 0; head < out.shape(1); ++head) {
                    for (U64 sample = 0; sample < out.shape(2); ++sample) {
                        const F32 sign = (sample & 1ULL) != 0 ? -1.0f : 1.0f;
                        for (U64 plane = 0; plane < out.shape(3); ++plane) {
                            REQUIRE(out.at<CF32>(batch, head, sample, plane) ==
                                    input.at(batch, head, sample, plane) * sign);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Invert Module - Trailing Batch Strided View And Attributes",
          "[modules][invert][batch][trailing][strided][attributes]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor storage;
            REQUIRE(storage.create(DeviceType::CPU, DataType::CF32, {3, 2, 2}) ==
                    Result::SUCCESS);
            for (U64 row = 0; row < 3; ++row) {
                for (U64 column = 0; column < 2; ++column) {
                    const F32 value = static_cast<F32>((row * 2) + column + 1);
                    storage.at<CF32>(row, column, 1) = CF32(value, -value);
                }
            }

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(), Token(), Token(1)}) == Result::SUCCESS);
            REQUIRE(input.shape() == Shape{3, 2});
            REQUIRE(input.offset() != 0);
            REQUIRE_FALSE(input.contiguous());
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("source", std::string("strided-view")) ==
                    Result::SUCCESS);

            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.shape() == input.shape());
            REQUIRE(out.dtype() == input.dtype());
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{0});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == 1);
            REQUIRE(out.hasAttribute("source"));
            REQUIRE(std::any_cast<std::string>(out.attribute("source")) == "strided-view");

            const F64 phaseStep = 2.0 * JST_PI *
                                  static_cast<F64>(input.shape(0) / 2) /
                                  static_cast<F64>(input.shape(0));
            for (U64 row = 0; row < out.shape(0); ++row) {
                const F64 phase = phaseStep * static_cast<F64>(row);
                const CF32 phasor{static_cast<F32>(std::cos(phase)),
                                  static_cast<F32>(std::sin(phase))};
                for (U64 column = 0; column < out.shape(1); ++column) {
                    const CF32 expected = input.at<CF32>(row, column) * phasor;
                    REQUIRE_THAT(out.at<CF32>(row, column).real(),
                                 Catch::Matchers::WithinAbs(expected.real(), 1e-5f));
                    REQUIRE_THAT(out.at<CF32>(row, column).imag(),
                                 Catch::Matchers::WithinAbs(expected.imag(), 1e-5f));
                }
            }
        }
    }
}

TEST_CASE("Invert Module - Missing Or Invalid Signal Metadata Error",
          "[modules][invert][batch][validation]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireInvertValidationError(impl, DataType::CF32, {2, 3});
            RequireInvertValidationError(
                impl, DataType::CF32, {2, 3}, I64{1});
            RequireInvertValidationError(
                impl, DataType::CF32, {2, 3}, Index{1}, I64{0});
            RequireInvertValidationError(
                impl, DataType::CF32, {2, 3}, Index{1}, Index{2});
            RequireInvertValidationError(
                impl, DataType::CF32, {3}, Index{0}, Index{0});
            RequireInvertValidationError(
                impl, DataType::CF32, {2, 3}, Index{1}, std::any{}, Index{1});
        }
    }
}
