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
            config.axis = 0;

            ctx.setConfig(config);

            // Buffer: [8] = {1,2,3,4,5,6,7,8}
            // Overlap: [3] = {10,20,30}
            // On first run, previousOverlap is zero.
            // Output = buffer with overlap[batch-1] added.
            // For 1D (rank=1), batch dim doesn't apply the same
            // way. The rank > 1 check means previousOverlap
            // shape = overlap shape for 1D.
            const U64 bufSize = 8;
            const U64 ovlSize = 3;

            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::F32,
                                  {bufSize}) == Result::SUCCESS);
            Tensor overlap;
            REQUIRE(overlap.create(DeviceType::CPU, DataType::F32,
                                   {ovlSize}) == Result::SUCCESS);

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

TEST_CASE("OverlapAdd - 2D CF32 Batched",
          "[modules][overlap_add][cf32]") {
    auto implementations =
        Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("overlap_add", impl.device,
                           impl.runtime, impl.provider);

            Modules::OverlapAdd config;
            config.axis = 1;

            ctx.setConfig(config);

            // Buffer: [2, 8] Overlap: [2, 3]
            const U64 batches = 2;
            const U64 bufCols = 8;
            const U64 ovlCols = 3;

            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::CF32,
                                  {batches, bufCols})
                    == Result::SUCCESS);
            Tensor overlap;
            REQUIRE(overlap.create(DeviceType::CPU, DataType::CF32,
                                   {batches, ovlCols})
                    == Result::SUCCESS);

            // Fill buffer with 1.0.
            for (U64 b = 0; b < batches; ++b) {
                for (U64 c = 0; c < bufCols; ++c) {
                    buffer.at<CF32>(b, c) = CF32(1.0f, 0.0f);
                }
            }

            // Fill overlap: batch 0 = (10,0), batch 1 = (20,0).
            for (U64 c = 0; c < ovlCols; ++c) {
                overlap.at<CF32>(0, c) = CF32(10.0f, 0.0f);
                overlap.at<CF32>(1, c) = CF32(20.0f, 0.0f);
            }

            ctx.setInput("buffer", buffer);
            ctx.setInput("overlap", overlap);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // Batch 0: adds previousOverlap (zeros).
            // out[0, 0:3] = 1.0 + 0.0 = 1.0
            // out[0, 3:8] = 1.0
            for (U64 c = 0; c < bufCols; ++c) {
                REQUIRE_THAT(out.at<CF32>(0, c).real(),
                    Catch::Matchers::WithinAbs(1.0f, 1e-5f));
            }

            // Batch 1: adds overlap from batch 0.
            // out[1, 0:3] = 1.0 + 10.0 = 11.0
            // out[1, 3:8] = 1.0
            for (U64 c = 0; c < ovlCols; ++c) {
                REQUIRE_THAT(out.at<CF32>(1, c).real(),
                    Catch::Matchers::WithinAbs(11.0f, 1e-5f));
            }
            for (U64 c = ovlCols; c < bufCols; ++c) {
                REQUIRE_THAT(out.at<CF32>(1, c).real(),
                    Catch::Matchers::WithinAbs(1.0f, 1e-5f));
            }
        }
    }
}

TEST_CASE("OverlapAdd - Negative Axis",
          "[modules][overlap_add][axis]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("overlap_add", impl.device,
                            impl.runtime, impl.provider);

            Modules::OverlapAdd config;
            config.axis = -1;
            ctx.setConfig(config);

            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::F32,
                                  {2, 8}) == Result::SUCCESS);
            Tensor overlap;
            REQUIRE(overlap.create(DeviceType::CPU, DataType::F32,
                                   {2, 3}) == Result::SUCCESS);
            for (U64 index = 0; index < buffer.size(); ++index) {
                buffer.data<F32>()[index] = 1.0f;
            }
            for (U64 column = 0; column < overlap.shape(1); ++column) {
                overlap.at<F32>(0, column) = 10.0f;
                overlap.at<F32>(1, column) = 20.0f;
            }
            ctx.setInput("buffer", buffer);
            ctx.setInput("overlap", overlap);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("buffer");
            REQUIRE(output.shape() == Shape{2, 8});
            for (U64 column = 0; column < output.shape(1); ++column) {
                REQUIRE(output.at<F32>(0, column) == 1.0f);
                REQUIRE(output.at<F32>(1, column) ==
                        (column < overlap.shape(1) ? 11.0f : 1.0f));
            }
        }
    }
}

TEST_CASE("OverlapAdd - Validation rejects out-of-range axes",
          "[modules][overlap_add][validation][axis]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const I64 axis : {I64{2}, I64{-3}}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                            << impl.runtime << " Axis: " << axis) {
                const Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 8});
                const Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 3});

                Modules::OverlapAdd config;
                config.axis = axis;
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }
        }
    }
}

TEST_CASE("OverlapAdd - Validation rejects the reserved batch axis",
          "[modules][overlap_add][validation][axis]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const I64 axis : {I64{0}, I64{-2}}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                            << impl.runtime << " Axis: " << axis) {
                const Tensor buffer = MakeOverlapAddTensor(
                    impl, DataType::F32, {2, 4});
                const Tensor overlap = MakeOverlapAddTensor(
                    impl, DataType::F32, {1, 4});

                Modules::OverlapAdd config;
                config.axis = axis;
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
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
            config.axis = 1;

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
            config.axis = 0;

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
                RequireOverlapAddValidationError(impl, config, buffer, overlap);
            }
        }
    }
}

TEST_CASE("OverlapAdd - Equal overlap boundary preserves state across computes",
          "[modules][overlap_add][state]") {
    const auto implementations = Registry::ListAvailableModules("overlap_add");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Tensor buffer = MakeOverlapAddTensor(impl, DataType::F32, {3});
            Tensor overlap = MakeOverlapAddTensor(impl, DataType::F32, {3});
            buffer.at<F32>(0) = 1.0f;
            buffer.at<F32>(1) = 2.0f;
            buffer.at<F32>(2) = 3.0f;
            overlap.at<F32>(0) = 10.0f;
            overlap.at<F32>(1) = 20.0f;
            overlap.at<F32>(2) = 30.0f;

            TensorMap inputs;
            inputs["buffer"].requested("test", "buffer");
            inputs["buffer"].tensor = buffer;
            inputs["overlap"].requested("test", "overlap");
            inputs["overlap"].tensor = overlap;

            Modules::OverlapAdd config;
            config.axis = 0;
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("overlap_add", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);
            REQUIRE(module->state() == Module::State::CREATED);
            REQUIRE(module->outputs().at("buffer").tensor.shape() == Shape{3});

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

            Tensor output = module->outputs().at("buffer").tensor;
            REQUIRE(output.at<F32>(0) == 1.0f);
            REQUIRE(output.at<F32>(1) == 2.0f);
            REQUIRE(output.at<F32>(2) == 3.0f);

            buffer.at<F32>(0) = 4.0f;
            buffer.at<F32>(1) = 5.0f;
            buffer.at<F32>(2) = 6.0f;
            overlap.at<F32>(0) = 40.0f;
            overlap.at<F32>(1) = 50.0f;
            overlap.at<F32>(2) = 60.0f;
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

            REQUIRE(output.at<F32>(0) == 14.0f);
            REQUIRE(output.at<F32>(1) == 25.0f);
            REQUIRE(output.at<F32>(2) == 36.0f);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
