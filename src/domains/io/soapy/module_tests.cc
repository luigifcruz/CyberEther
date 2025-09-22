#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/io/soapy/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Soapy module rejects invalid batch dimensions",
          "[modules][soapy][validation]") {
    auto implementations = Registry::ListAvailableModules("soapy");
    if (implementations.empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        SECTION("numberOfBatches must be > 0") {
            TestContext ctx("soapy", impl.device, impl.runtime, impl.provider);

            Modules::Soapy config;
            config.numberOfBatches = 0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("numberOfTimeSamples must be > 0") {
            TestContext ctx("soapy", impl.device, impl.runtime, impl.provider);

            Modules::Soapy config;
            config.numberOfTimeSamples = 0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("bufferMultiplier must be > 0") {
            TestContext ctx("soapy", impl.device, impl.runtime, impl.provider);

            Modules::Soapy config;
            config.bufferMultiplier = 0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("non-default params still validate dimensions") {
            TestContext ctx("soapy", impl.device, impl.runtime, impl.provider);

            Modules::Soapy config;
            config.deviceString = "driver=mock";
            config.streamString = "bufflen=4096";
            config.frequency = 100.5e6f;
            config.sampleRate = 1.5e6f;
            config.automaticGain = false;
            config.numberOfBatches = 0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
