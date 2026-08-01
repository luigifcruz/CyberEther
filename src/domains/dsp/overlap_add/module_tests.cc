#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <memory>
#include <unordered_set>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/overlap_add/module.hh"

using namespace Jetstream;

namespace {

Tensor MakeOverlapAddTensor(const Registry::ModuleRegistration& impl,
                            const DataType dtype,
                            const Shape& shape) {
    Tensor tensor;
    REQUIRE(tensor.create(impl.device, dtype, shape) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("sampleAxis", static_cast<Index>(shape.size() - 1)) ==
            Result::SUCCESS);
    return tensor;
}

void RequireOverlapAddValidationError(const Registry::ModuleRegistration& impl,
                                      const Modules::OverlapAdd& config,
                                      const Tensor& buffer,
                                      const Tensor& overlap) {
    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = buffer;
    inputs["overlap"].requested("test", "overlap");
    inputs["overlap"].tensor = overlap;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("overlap_add", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("OverlapAdd - 1D F32 Basic",
          "[modules][overlap_add][f32]") {
    auto implementations =
        Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("overlap_add", impl.device,
                           impl.runtime, impl.provider);

            Modules::OverlapAdd config;

            ctx.setConfig(config);

            // Buffer: [8] = {1,2,3,4,5,6,7,8}
            // Overlap: [3] = {10,20,30}
            // On first run, previousOverlap is zero, so output is unchanged.
            const U64 bufSize = 8;
            const U64 ovlSize = 3;

            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::F32,
                                  {bufSize}) == Result::SUCCESS);
            Tensor overlap;
            REQUIRE(overlap.create(DeviceType::CPU, DataType::F32,
                                   {ovlSize}) == Result::SUCCESS);
            REQUIRE(buffer.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(overlap.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            for (U64 i = 0; i < bufSize; ++i) {
                buffer.at<F32>(i) = static_cast<F32>(i + 1);
            }
            for (U64 i = 0; i < ovlSize; ++i) {
                overlap.at<F32>(i) = static_cast<F32>((i + 1) * 10);
            }

            ctx.setInput("buffer", buffer);
            ctx.setInput("overlap", overlap);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // First run: previousOverlap is zeros.
            // For 1D, coords[0] == 0 always -> adds prevOverlap
            // (zeros).
            // So output = buffer unchanged.
            REQUIRE_THAT(out.at<F32>(0),
                Catch::Matchers::WithinAbs(1.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(1),
                Catch::Matchers::WithinAbs(2.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(2),
                Catch::Matchers::WithinAbs(3.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(7),
                Catch::Matchers::WithinAbs(8.0f, 1e-5f));
        }
    }
}

TEST_CASE("OverlapAdd - 3D CF32 Batched Heads",
          "[modules][overlap_add][cf32][batch][heads]") {
    auto implementations =
        Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("overlap_add", impl.device,
                           impl.runtime, impl.provider);

            Modules::OverlapAdd config;

            ctx.setConfig(config);

            const U64 batches = 2;
            const U64 heads = 2;
            const U64 bufferSamples = 8;
            const U64 overlapSamples = 3;

            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::CF32,
                                  {batches, heads, bufferSamples})
                     == Result::SUCCESS);
            Tensor overlap;
            REQUIRE(overlap.create(DeviceType::CPU, DataType::CF32,
                                   {batches, heads, overlapSamples})
                     == Result::SUCCESS);
            REQUIRE(buffer.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(overlap.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(buffer.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(overlap.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(buffer.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(overlap.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);

            for (U64 batch = 0; batch < batches; ++batch) {
                for (U64 head = 0; head < heads; ++head) {
                    for (U64 sample = 0; sample < bufferSamples; ++sample) {
                        buffer.at<CF32>(batch, head, sample) = CF32(1.0f, 0.0f);
                    }
                }
            }

            for (U64 batch = 0; batch < batches; ++batch) {
                for (U64 head = 0; head < heads; ++head) {
                    const F32 value = static_cast<F32>(10 + 20 * batch + 10 * head);
                    for (U64 sample = 0; sample < overlapSamples; ++sample) {
                        overlap.at<CF32>(batch, head, sample) = CF32(value, 0.0f);
                    }
                }
            }

            ctx.setInput("buffer", buffer);
            ctx.setInput("overlap", overlap);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape() == Shape{batches, heads, bufferSamples});
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{2});
            REQUIRE(out.hasAttribute("batchAxis"));
            REQUIRE(out.attribute("batchAxis").type() == typeid(Index));
            REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(out.attribute("channelAxis")) == Index{1});

            for (U64 head = 0; head < heads; ++head) {
                for (U64 sample = 0; sample < bufferSamples; ++sample) {
                    REQUIRE_THAT(out.at<CF32>(0, head, sample).real(),
                        Catch::Matchers::WithinAbs(1.0f, 1e-5f));
                    const F32 expected = sample < overlapSamples
                        ? static_cast<F32>(11 + 10 * head)
                        : 1.0f;
                    REQUIRE_THAT(out.at<CF32>(1, head, sample).real(),
                        Catch::Matchers::WithinAbs(expected, 1e-5f));
                }
            }
        }
    }
}

TEST_CASE("OverlapAdd - Trailing Batch Metadata",
           "[modules][overlap_add][batch]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("overlap_add", impl.device,
                            impl.runtime, impl.provider);

            Modules::OverlapAdd config;
            ctx.setConfig(config);

            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::F32,
                                  {8, 2}) == Result::SUCCESS);
            Tensor overlap;
            REQUIRE(overlap.create(DeviceType::CPU, DataType::F32,
                                   {3, 2}) == Result::SUCCESS);
            REQUIRE(buffer.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(overlap.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(buffer.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(overlap.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            for (U64 index = 0; index < buffer.size(); ++index) {
                buffer.data<F32>()[index] = 1.0f;
            }
            for (U64 sample = 0; sample < overlap.shape(0); ++sample) {
                overlap.at<F32>(sample, 0) = 10.0f;
                overlap.at<F32>(sample, 1) = 20.0f;
            }
            ctx.setInput("buffer", buffer);
            ctx.setInput("overlap", overlap);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("buffer");
            REQUIRE(output.shape() == Shape{8, 2});
            REQUIRE(output.hasAttribute("batchAxis"));
            REQUIRE(output.attribute("batchAxis").type() == typeid(Index));
            REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == Index{1});
            REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{0});
            for (U64 sample = 0; sample < output.shape(0); ++sample) {
                REQUIRE(output.at<F32>(sample, 0) == 1.0f);
                REQUIRE(output.at<F32>(sample, 1) ==
                        (sample < overlap.shape(0) ? 11.0f : 1.0f));
            }
        }
    }
}

TEST_CASE("OverlapAdd - Validation rejects malformed signal metadata",
           "[modules][overlap_add][validation][metadata]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Modules::OverlapAdd config;

            SECTION("sample axis is required") {
                Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 8});
                Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 3});
                REQUIRE(buffer.removeAttribute("sampleAxis") == Result::SUCCESS);
                REQUIRE(overlap.removeAttribute("sampleAxis") == Result::SUCCESS);
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }

            SECTION("sample axis type must be exact") {
                Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 8});
                Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 3});
                REQUIRE(buffer.setAttribute("sampleAxis", I64{1}) == Result::SUCCESS);
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }

            SECTION("batch axis type must be exact") {
                Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 8});
                Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 3});
                REQUIRE(buffer.setAttribute("batchAxis", I64{0}) == Result::SUCCESS);
                REQUIRE(overlap.setAttribute("batchAxis", I64{0}) == Result::SUCCESS);
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }

            SECTION("batch axis must be in range") {
                Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 8});
                Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 3});
                REQUIRE(buffer.setAttribute("batchAxis", Index{2}) == Result::SUCCESS);
                REQUIRE(overlap.setAttribute("batchAxis", Index{2}) == Result::SUCCESS);
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }

            SECTION("signal roles must be distinct") {
                Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {8});
                Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {3});
                REQUIRE(buffer.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
                REQUIRE(overlap.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }
        }
    }
}

TEST_CASE("OverlapAdd - Validation rejects mismatched signal roles",
           "[modules][overlap_add][validation][metadata]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            SECTION("only one input is tagged") {
                Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 4});
                const Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 4});
                REQUIRE(buffer.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
                RequireOverlapAddValidationError(
                    impl, Modules::OverlapAdd{}, buffer, overlap);
            }

            SECTION("batch axes differ") {
                Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 2, 4});
                Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 2, 4});
                REQUIRE(buffer.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
                REQUIRE(overlap.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
                RequireOverlapAddValidationError(
                    impl, Modules::OverlapAdd{}, buffer, overlap);
            }

            SECTION("sample axes differ") {
                Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {4, 4});
                Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {4, 4});
                REQUIRE(buffer.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
                RequireOverlapAddValidationError(
                    impl, Modules::OverlapAdd{}, buffer, overlap);
            }

            SECTION("channel axes differ") {
                Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 2, 4});
                Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 2, 4});
                REQUIRE(buffer.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);
                REQUIRE(overlap.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
                RequireOverlapAddValidationError(
                    impl, Modules::OverlapAdd{}, buffer, overlap);
            }
        }
    }
}

TEST_CASE("OverlapAdd - Validation rejects rank, shape, and extent mismatches",
          "[modules][overlap_add][validation][shape]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Modules::OverlapAdd config;

            SECTION("ranks must match") {
                const Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 4});
                const Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2});
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }

            SECTION("non-overlap dimensions must match") {
                const Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 4});
                const Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {3, 2});
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }

            SECTION("overlap extent must not exceed buffer extent") {
                const Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 2});
                const Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 3});
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }
        }
    }
}

TEST_CASE("OverlapAdd - CPU validation rejects dtype and allocation errors",
          "[modules][overlap_add][validation][cpu]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Modules::OverlapAdd config;

            SECTION("input dtypes must match exactly") {
                const Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {4});
                const Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::CF32, {2});
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }

            SECTION("input dtype must be supported") {
                const Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F64, {4});
                const Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F64, {2});
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }

            SECTION("allocation alignment must be representable") {
                F32 storage = 0.0f;
                const U64 extent = std::numeric_limits<U64>::max() / sizeof(F32);
                Tensor buffer;
                Tensor overlap;
                REQUIRE(buffer.create(&storage, DeviceType::CPU, DataType::F32,
                                      {extent}) == Result::SUCCESS);
                REQUIRE(overlap.create(&storage, DeviceType::CPU, DataType::F32,
                                       {1}) == Result::SUCCESS);
                REQUIRE(buffer.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
                REQUIRE(overlap.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }
        }
    }
}

TEST_CASE("OverlapAdd - Channels preserve independent state across computes",
           "[modules][overlap_add][state][channels]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Tensor buffer = MakeOverlapAddTensor(impl, DataType::F32, {2, 3});
            Tensor overlap = MakeOverlapAddTensor(impl, DataType::F32, {2, 3});
            REQUIRE(buffer.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(overlap.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);
            for (U64 sample = 0; sample < 3; ++sample) {
                buffer.at<F32>(0, sample) = static_cast<F32>(sample + 1);
                buffer.at<F32>(1, sample) = static_cast<F32>(sample + 4);
                overlap.at<F32>(0, sample) = static_cast<F32>(10 * (sample + 1));
                overlap.at<F32>(1, sample) = static_cast<F32>(100 * (sample + 1));
            }

            TensorMap inputs;
            inputs["buffer"].requested("test", "buffer");
            inputs["buffer"].tensor = buffer;
            inputs["overlap"].requested("test", "overlap");
            inputs["overlap"].tensor = overlap;

            Modules::OverlapAdd config;
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("overlap_add", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);
            REQUIRE(module->state() == Module::State::CREATED);
            REQUIRE(module->outputs().at("buffer").tensor.shape() == Shape{2, 3});
            REQUIRE(std::any_cast<Index>(module->outputs().at("buffer").tensor.attribute(
                        "sampleAxis")) == Index{1});
            REQUIRE(std::any_cast<Index>(module->outputs().at("buffer").tensor.attribute(
                        "channelAxis")) == Index{0});
            REQUIRE_FALSE(module->outputs().at("buffer").tensor.hasAttribute(
                "batchAxis"));

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

            Tensor output = module->outputs().at("buffer").tensor;
            for (U64 head = 0; head < 2; ++head) {
                for (U64 sample = 0; sample < 3; ++sample) {
                    REQUIRE(output.at<F32>(head, sample) ==
                            buffer.at<F32>(head, sample));
                }
            }

            for (U64 sample = 0; sample < 3; ++sample) {
                buffer.at<F32>(0, sample) = static_cast<F32>(sample + 7);
                buffer.at<F32>(1, sample) = static_cast<F32>(sample + 10);
                overlap.at<F32>(0, sample) = static_cast<F32>(40 + 10 * sample);
                overlap.at<F32>(1, sample) = static_cast<F32>(400 + 100 * sample);
            }
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

            for (U64 sample = 0; sample < 3; ++sample) {
                REQUIRE(output.at<F32>(0, sample) ==
                        static_cast<F32>(17 + 11 * sample));
                REQUIRE(output.at<F32>(1, sample) ==
                        static_cast<F32>(110 + 101 * sample));
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
