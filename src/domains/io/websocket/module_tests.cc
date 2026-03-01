#include <catch2/catch_test_macros.hpp>

#include <array>

#include "jetstream/domains/io/websocket/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Websocket module rejects invalid configuration",
          "[modules][websocket][validation]") {
    auto implementations = Registry::ListAvailableModules("websocket");
    if (implementations.empty()) {
        SUCCEED("Websocket module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        SECTION("numberOfBatches must be > 0") {
            TestContext ctx("websocket", impl.device, impl.runtime,
                            impl.provider);
            Modules::Websocket config;
            config.numberOfBatches = 0;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("numberOfTimeSamples must be > 0") {
            TestContext ctx("websocket", impl.device, impl.runtime,
                            impl.provider);
            Modules::Websocket config;
            config.numberOfTimeSamples = 0;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("bufferMultiplier must be > 0") {
            TestContext ctx("websocket", impl.device, impl.runtime,
                            impl.provider);
            Modules::Websocket config;
            config.bufferMultiplier = 0;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("dataType must be valid") {
            TestContext ctx("websocket", impl.device, impl.runtime,
                            impl.provider);
            Modules::Websocket config;
            config.dataType = "I32";
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("url must be non-empty") {
            TestContext ctx("websocket", impl.device, impl.runtime,
                            impl.provider);
            Modules::Websocket config;
            config.url = "";
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::INCOMPLETE);
        }

        SECTION("supported data types pass validation") {
            const std::array<std::string, 5> validTypes = {
                "CF32", "CU8", "CS8", "CI16", "CU16"
            };

            for (const auto& dataType : validTypes) {
                DYNAMIC_SECTION("type=" << dataType) {
                    TestContext ctx("websocket", impl.device, impl.runtime,
                                    impl.provider);
                    Modules::Websocket config;
                    config.url = "";
                    config.dataType = dataType;
                    config.numberOfBatches = 2;
                    config.numberOfTimeSamples = 64;
                    config.bufferMultiplier = 2;
                    ctx.setConfig(config);

                    REQUIRE(ctx.run() == Result::INCOMPLETE);
                }
            }
        }
    }
}
