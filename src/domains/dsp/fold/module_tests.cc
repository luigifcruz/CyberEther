#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <string>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/fold/module.hh"

using namespace Jetstream;

namespace {

Tensor MakeFoldTensor(const Registry::ModuleRegistration& impl,
                      const DataType dtype,
                      const Shape& shape,
                      const bool broadcast = false) {
    Tensor tensor;
    if (broadcast) {
        REQUIRE(tensor.create(impl.device, dtype, Shape(shape.size(), 1)) == Result::SUCCESS);
        REQUIRE(tensor.broadcastTo(shape) == Result::SUCCESS);
    } else if (shape.empty()) {
        REQUIRE(tensor.create(impl.device, dtype, {1}) == Result::SUCCESS);
        REQUIRE(tensor.squeezeDims(0) == Result::SUCCESS);
    } else {
        REQUIRE(tensor.create(impl.device, dtype, shape) == Result::SUCCESS);
    }
    return tensor;
}

void RequireFoldValidationError(const Registry::ModuleRegistration& impl,
                                const Modules::Fold& config,
                                const Tensor& input) {
    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("fold", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Fold - 1D CF32 Uniform", "[modules][fold][cf32]") {
    auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("fold", impl.device,
                           impl.runtime, impl.provider);

            Modules::Fold config;
            config.offset = 0;
            config.size = 4;

            ctx.setConfig(config);

            // Create input: 16 elements all set to 1.0.
            const U64 inputSize = 16;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {inputSize}) == Result::SUCCESS);

            for (U64 i = 0; i < inputSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // Decimation factor = 16 / 4 = 4.
            // Each output element accumulates 4 inputs of 1.0,
            // then divides by 4 -> 1.0.
            for (U64 i = 0; i < config.size; ++i) {
                REQUIRE_THAT(out.at<CF32>(i).real(),
                    Catch::Matchers::WithinAbs(1.0f, 1e-5f));
                REQUIRE_THAT(out.at<CF32>(i).imag(),
                    Catch::Matchers::WithinAbs(0.0f, 1e-5f));
            }
        }
    }
}

TEST_CASE("Fold - 1D F32 Ramp", "[modules][fold][f32]") {
    auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("fold", impl.device,
                           impl.runtime, impl.provider);

            Modules::Fold config;
            config.offset = 0;
            config.size = 4;

            ctx.setConfig(config);

            // Create input: 8 elements [0,1,2,3,4,5,6,7].
            const U64 inputSize = 8;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32,
                                 {inputSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            for (U64 i = 0; i < inputSize; ++i) {
                input.at<F32>(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // Decimation factor = 8 / 4 = 2.
            // out[0] = (0 + 4) / 2 = 2.0
            // out[1] = (1 + 5) / 2 = 3.0
            // out[2] = (2 + 6) / 2 = 4.0
            // out[3] = (3 + 7) / 2 = 5.0
            REQUIRE_THAT(out.at<F32>(0),
                Catch::Matchers::WithinAbs(2.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(1),
                Catch::Matchers::WithinAbs(3.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(2),
                Catch::Matchers::WithinAbs(4.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(3),
                Catch::Matchers::WithinAbs(5.0f, 1e-5f));
        }
    }
}

TEST_CASE("Fold - Avoids intermediate overflow while averaging",
          "[modules][fold][f32][numeric]") {
    const auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fold", impl.device, impl.runtime, impl.provider);
            Modules::Fold config;
            config.size = 1;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            input.at<F32>(0) = std::numeric_limits<F32>::max();
            input.at<F32>(1) = std::numeric_limits<F32>::max();
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("buffer");
            REQUIRE(std::isfinite(output.at<F32>(0)));
            REQUIRE(output.at<F32>(0) == std::numeric_limits<F32>::max());
        }
    }
}

TEST_CASE("Fold - 1D F32 With Offset", "[modules][fold][offset]") {
    auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("fold", impl.device,
                           impl.runtime, impl.provider);

            Modules::Fold config;
            config.offset = 2;
            config.size = 4;

            ctx.setConfig(config);

            // Create input: 8 elements [0,1,2,3,4,5,6,7].
            const U64 inputSize = 8;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32,
                                 {inputSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            for (U64 i = 0; i < inputSize; ++i) {
                input.at<F32>(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // With offset=2, before folding each index is shifted:
            // idx 0 -> (0+2)%8=2 -> 2%4=2
            // idx 1 -> (1+2)%8=3 -> 3%4=3
            // idx 2 -> (2+2)%8=4 -> 4%4=0
            // idx 3 -> (3+2)%8=5 -> 5%4=1
            // idx 4 -> (4+2)%8=6 -> 6%4=2
            // idx 5 -> (5+2)%8=7 -> 7%4=3
            // idx 6 -> (6+2)%8=0 -> 0%4=0
            // idx 7 -> (7+2)%8=1 -> 1%4=1
            // out[0] = (2 + 6) / 2 = 4.0
            // out[1] = (3 + 7) / 2 = 5.0
            // out[2] = (0 + 4) / 2 = 2.0
            // out[3] = (1 + 5) / 2 = 3.0
            REQUIRE_THAT(out.at<F32>(0),
                Catch::Matchers::WithinAbs(4.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(1),
                Catch::Matchers::WithinAbs(5.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(2),
                Catch::Matchers::WithinAbs(2.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(3),
                Catch::Matchers::WithinAbs(3.0f, 1e-5f));
        }
    }
}

TEST_CASE("Fold - 2D F32 Heads Fold Independently", "[modules][fold][heads]") {
    auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("fold", impl.device,
                           impl.runtime, impl.provider);

            Modules::Fold config;
            config.offset = 0;
            config.size = 4;

            ctx.setConfig(config);

            // Create input shape [2, 8]:
            // row0 = [0..7], row1 = [10..17]
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32,
                                  {2, 8}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);
            for (U64 i = 0; i < 8; ++i) {
                input.at<F32>(0, i) = static_cast<F32>(i);
                input.at<F32>(1, i) = static_cast<F32>(10 + i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 4);
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{1});
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{0});
            REQUIRE_FALSE(out.hasAttribute("batchAxis"));

            // Decimation factor = 8 / 4 = 2 on the sample axis.
            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(0, i),
                    Catch::Matchers::WithinAbs(static_cast<F32>(2 + i), 1e-5f));
                REQUIRE_THAT(out.at<F32>(1, i),
                    Catch::Matchers::WithinAbs(static_cast<F32>(12 + i), 1e-5f));
            }
        }
    }
}

TEST_CASE("Fold - 4D F32 Batched Heads With Opaque Planes",
          "[modules][fold][batch][heads][opaque]") {
    const auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("fold", impl.device, impl.runtime, impl.provider);

            Modules::Fold config;
            config.size = 4;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32,
                                 {2, 2, 8, 2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("layout", std::string("batched-heads")) ==
                    Result::SUCCESS);
            for (U64 batch = 0; batch < 2; ++batch) {
                for (U64 head = 0; head < 2; ++head) {
                    for (U64 sample = 0; sample < 8; ++sample) {
                        for (U64 plane = 0; plane < 2; ++plane) {
                            input.at<F32>(batch, head, sample, plane) =
                                static_cast<F32>(1000 * batch + 100 * head +
                                                 10 * plane + sample);
                        }
                    }
                }
            }
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("buffer");
            REQUIRE(output.shape() == Shape{2, 2, 4, 2});
            REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{2});
            REQUIRE(output.hasAttribute("batchAxis"));
            REQUIRE(output.attribute("batchAxis").type() == typeid(Index));
            REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});
            REQUIRE(std::any_cast<std::string>(output.attribute("layout")) ==
                    "batched-heads");
            for (U64 batch = 0; batch < 2; ++batch) {
                for (U64 head = 0; head < 2; ++head) {
                    for (U64 sample = 0; sample < 4; ++sample) {
                        for (U64 plane = 0; plane < 2; ++plane) {
                            REQUIRE_THAT(output.at<F32>(batch, head, sample, plane),
                                Catch::Matchers::WithinAbs(static_cast<F32>(
                                    1000 * batch + 100 * head + 10 * plane +
                                    2 + sample), 1e-5f));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Fold - 2D F32 With Trailing Batch", "[modules][fold][batch]") {
    const auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("fold", impl.device, impl.runtime, impl.provider);

            Modules::Fold config;
            config.size = 4;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32,
                                  {8, 2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            for (U64 i = 0; i < 8; ++i) {
                input.at<F32>(i, 0) = static_cast<F32>(i);
                input.at<F32>(i, 1) = static_cast<F32>(10 + i);
            }
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& out = ctx.output("buffer");
            REQUIRE(out.shape() == Shape{4, 2});
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{0});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(out.attribute("batchAxis").type() == typeid(Index));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});
            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(i, 0),
                    Catch::Matchers::WithinAbs(static_cast<F32>(2 + i), 1e-5f));
                REQUIRE_THAT(out.at<F32>(i, 1),
                    Catch::Matchers::WithinAbs(static_cast<F32>(12 + i), 1e-5f));
            }
        }
    }
}

TEST_CASE("Fold - Validation rejects missing or malformed signal metadata",
          "[modules][fold][validation][metadata]") {
    const auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Fold config;
            config.size = 2;

            SECTION("untagged multidimensional input") {
                RequireFoldValidationError(
                    impl, config, MakeFoldTensor(impl, DataType::F32, {2, 4}));
            }

            SECTION("batch axis type must be exact") {
                Tensor input = MakeFoldTensor(impl, DataType::F32, {2, 4});
                REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
                REQUIRE(input.setAttribute("batchAxis", I64{0}) == Result::SUCCESS);
                RequireFoldValidationError(impl, config, input);
            }

            SECTION("batch axis must be in range") {
                Tensor input = MakeFoldTensor(impl, DataType::F32, {2, 4});
                REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
                REQUIRE(input.setAttribute("batchAxis", Index{2}) == Result::SUCCESS);
                RequireFoldValidationError(impl, config, input);
            }

            SECTION("signal roles must not share an axis") {
                Tensor input = MakeFoldTensor(impl, DataType::F32, {4});
                REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
                REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
                RequireFoldValidationError(impl, config, input);
            }

            SECTION("sample axis type must be exact") {
                Tensor input = MakeFoldTensor(impl, DataType::F32, {2, 4});
                REQUIRE(input.setAttribute("sampleAxis", I64{1}) == Result::SUCCESS);
                RequireFoldValidationError(impl, config, input);
            }

            SECTION("sample and channel roles must not share an axis") {
                Tensor input = MakeFoldTensor(impl, DataType::F32, {2, 4});
                REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
                REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
                RequireFoldValidationError(impl, config, input);
            }
        }
    }
}

TEST_CASE("Fold - Validation rejects invalid size and offset",
          "[modules][fold][validation][config]") {
    const auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor input = MakeFoldTensor(impl, DataType::F32, {8});
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            Modules::Fold config;

            config.size = 0;
            RequireFoldValidationError(impl, config, input);

            config.size = 3;
            RequireFoldValidationError(impl, config, input);

            config.size = 4;
            config.offset = 9;
            RequireFoldValidationError(impl, config, input);
        }
    }
}

TEST_CASE("Fold - Validation rejects rank zero and unsupported dtype",
          "[modules][fold][validation]") {
    const auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Fold config;
            config.size = 1;
            const Tensor scalar = MakeFoldTensor(impl, DataType::F32, {});
            RequireFoldValidationError(impl, config, scalar);

            Tensor f64 = MakeFoldTensor(impl, DataType::F64, {8});
            REQUIRE(f64.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            config.size = 4;
            RequireFoldValidationError(impl, config, f64);
        }
    }
}

TEST_CASE("Fold - CPU validation rejects unsupported allocation size",
          "[modules][fold][validation][allocation]") {
    const auto implementations = Registry::ListAvailableModules("fold");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const U64 size = std::numeric_limits<U64>::max() / sizeof(F32);
            Tensor input = MakeFoldTensor(impl, DataType::F32, {size}, true);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            Modules::Fold config;
            config.size = size;
            RequireFoldValidationError(impl, config, input);
        }
    }
}
