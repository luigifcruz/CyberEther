#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/io/audio/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Audio module rejects non-positive input sample rate",
          "[modules][audio][validation]") {
    auto implementations = Registry::ListAvailableModules("audio");
    if (implementations.empty()) {
        SUCCEED("Audio module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("audio", impl.device, impl.runtime, impl.provider);

            Modules::Audio config;
            config.inSampleRate = 0.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({64});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Audio module rejects non-positive output sample rate",
          "[modules][audio][validation]") {
    auto implementations = Registry::ListAvailableModules("audio");
    if (implementations.empty()) {
        SUCCEED("Audio module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("audio", impl.device, impl.runtime, impl.provider);

            Modules::Audio config;
            config.outSampleRate = -1.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({64});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
