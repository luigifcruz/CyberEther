#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/overlap_add/module.hh"

using namespace Jetstream;

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

TEST_CASE("OverlapAdd - Rank Mismatch Error",
          "[modules][overlap_add][error]") {
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

            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::F32,
                                  {4, 8}) == Result::SUCCESS);
            Tensor overlap;
            REQUIRE(overlap.create(DeviceType::CPU, DataType::F32,
                                   {3}) == Result::SUCCESS);

            ctx.setInput("buffer", buffer);
            ctx.setInput("overlap", overlap);

            REQUIRE(ctx.run() != Result::SUCCESS);
        }
    }
}

TEST_CASE("OverlapAdd - Non-Axis Shape Mismatch Error",
          "[modules][overlap_add][error]") {
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

            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::F32,
                                  {2, 8}) == Result::SUCCESS);
            Tensor overlap;
            REQUIRE(overlap.create(DeviceType::CPU, DataType::F32,
                                   {3, 3}) == Result::SUCCESS);

            ctx.setInput("buffer", buffer);
            ctx.setInput("overlap", overlap);

            REQUIRE(ctx.run() != Result::SUCCESS);
        }
    }
}

TEST_CASE("OverlapAdd - Dtype Mismatch Error",
          "[modules][overlap_add][error]") {
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

            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::F32,
                                  {2, 8}) == Result::SUCCESS);
            Tensor overlap;
            REQUIRE(overlap.create(DeviceType::CPU, DataType::CF32,
                                   {2, 3}) == Result::SUCCESS);

            ctx.setInput("buffer", buffer);
            ctx.setInput("overlap", overlap);

            REQUIRE(ctx.run() != Result::SUCCESS);
        }
    }
}
