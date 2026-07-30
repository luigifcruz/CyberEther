#include <catch2/catch_test_macros.hpp>

#include <any>
#include <array>
#include <limits>
#include <string>

#include "jetstream/domains/io/websocket/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "module_impl.hh"
#include "test_server.hh"

using namespace Jetstream;

namespace {

void RequireWebsocketValidationResult(const Registry::ModuleRegistration& impl,
                                       const Modules::Websocket& config,
                                       const Result expected) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("websocket", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);

    Result result = Result::SUCCESS;
    REQUIRE_NOTHROW(result = module->create("test", config, {}));
    REQUIRE(result == expected);
    REQUIRE(module->state() == (expected == Result::INCOMPLETE
                                    ? Module::State::INCOMPLETE
                                    : Module::State::ERRORED));
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());

    const auto& retained =
        static_cast<const Modules::Websocket&>(module->config());
    REQUIRE(retained.url == Modules::Websocket{}.url);
    REQUIRE(retained.dataType == Modules::Websocket{}.dataType);
    REQUIRE(retained.numberOfBatches == Modules::Websocket{}.numberOfBatches);
    REQUIRE(retained.numberOfTimeSamples ==
            Modules::Websocket{}.numberOfTimeSamples);
    REQUIRE(retained.bufferMultiplier == Modules::Websocket{}.bufferMultiplier);
}

void RequireWebsocketValidationSuccess(const Registry::ModuleRegistration& impl,
                                        const Modules::Websocket& config) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("websocket", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);

    auto* websocket = module->getImpl<Modules::WebsocketImpl>();
    REQUIRE(websocket != nullptr);
    *websocket->candidate() = config;
    REQUIRE(websocket->validate() == Result::SUCCESS);
    REQUIRE(module->state() == Module::State::NONE);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Websocket module rejects invalid configuration during validation",
          "[modules][websocket][validation][phase]") {
    const auto implementations = Registry::ListAvailableModules("websocket");
    if (implementations.empty()) {
        SUCCEED("Websocket module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        SECTION("numberOfBatches must be > 0") {
            Modules::Websocket config;
            config.numberOfBatches = 0;
            RequireWebsocketValidationResult(impl, config, Result::ERROR);
        }

        SECTION("numberOfTimeSamples must be > 0") {
            Modules::Websocket config;
            config.numberOfTimeSamples = 0;
            RequireWebsocketValidationResult(impl, config, Result::ERROR);
        }

        SECTION("bufferMultiplier must be > 0") {
            Modules::Websocket config;
            config.bufferMultiplier = 0;
            RequireWebsocketValidationResult(impl, config, Result::ERROR);
        }

        SECTION("dataType must be valid") {
            Modules::Websocket config;
            config.dataType = "CS8";
            RequireWebsocketValidationResult(impl, config, Result::ERROR);
        }

        SECTION("url must be non-empty") {
            Modules::Websocket config;
            config.url = "";
            RequireWebsocketValidationResult(impl, config, Result::INCOMPLETE);
        }

        SECTION("url syntax must be valid") {
            for (const std::string& url : {
                     "http://localhost:8765",
                     "ws://localhost:8765/feed#fragment",
                     "ws:///feed",
                 }) {
                DYNAMIC_SECTION(url) {
                    Modules::Websocket config;
                    config.url = url;
                    RequireWebsocketValidationResult(impl, config, Result::ERROR);
                }
            }
        }

#ifndef JST_OS_BROWSER
        SECTION("native URL syntax must be valid") {
            for (const std::string& url : {
                     "ws://:8765/feed",
                     "ws://localhost:not-a-port/feed",
                     "ws://[2001:db8::1/feed",
                 }) {
                DYNAMIC_SECTION(url) {
                    Modules::Websocket config;
                    config.url = url;
                    RequireWebsocketValidationResult(impl, config, Result::ERROR);
                }
            }
        }
#endif

        SECTION("output dimensions must not overflow") {
            Modules::Websocket config;
            config.numberOfBatches = std::numeric_limits<U64>::max();
            config.numberOfTimeSamples = 2;
            RequireWebsocketValidationResult(impl, config, Result::ERROR);
        }

        SECTION("output byte size must not overflow") {
            Modules::Websocket config;
            config.dataType = "CF32";
            config.numberOfBatches = std::numeric_limits<U64>::max() /
                                     DataTypeSize(DataType::CF32) + 1;
            config.numberOfTimeSamples = 1;
            config.bufferMultiplier = 1;
            RequireWebsocketValidationResult(impl, config, Result::ERROR);
        }

        SECTION("circular buffer size must not overflow") {
            Modules::Websocket config;
            config.dataType = "U8";
            config.numberOfBatches = std::numeric_limits<U64>::max() / 2 + 1;
            config.numberOfTimeSamples = 1;
            config.bufferMultiplier = 2;
            RequireWebsocketValidationResult(impl, config, Result::ERROR);
        }

        SECTION("output allocation must be representable") {
            Modules::Websocket config;
            config.dataType = "U8";
            config.numberOfBatches = std::numeric_limits<std::size_t>::max();
            config.numberOfTimeSamples = 1;
            config.bufferMultiplier = 1;
            RequireWebsocketValidationResult(impl, config, Result::ERROR);
        }
    }
}

TEST_CASE("Websocket supported dtypes validate without network access",
          "[modules][websocket][validation]") {
    const auto implementations = Registry::ListAvailableModules("websocket");
    if (implementations.empty()) {
        SUCCEED("Websocket module is unavailable in this build.");
        return;
    }

    const std::array<std::string, 10> validTypes = {
        "CF32", "F32", "CI8", "I8", "CU8",
        "U8", "CI16", "I16", "CU16", "U16",
    };
    for (const auto& impl : implementations) {
        for (const auto& dataType : validTypes) {
            DYNAMIC_SECTION("type=" << dataType << " device=" << impl.device) {
                Modules::Websocket config;
                config.dataType = dataType;
                config.numberOfBatches = 2;
                config.numberOfTimeSamples = 64;
                config.bufferMultiplier = 2;
                RequireWebsocketValidationSuccess(impl, config);
            }
        }
    }
}

#ifndef JST_OS_BROWSER
TEST_CASE("Websocket module rolls back native URL validation rejection",
          "[modules][websocket][validation][reconfigure]") {
    Tests::WebsocketTestServer server;
    REQUIRE(server.valid());

    const auto implementations = Registry::ListAvailableModules("websocket");
    REQUIRE(!implementations.empty());
    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("device=" << impl.device << " runtime=" << impl.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("websocket", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);

            Modules::Websocket config;
            config.url = server.url();
            config.dataType = "U8";
            config.numberOfBatches = 1;
            config.numberOfTimeSamples = 1;
            config.bufferMultiplier = 1;
            REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
            const auto& output = module->outputs().at("signal").tensor;
            REQUIRE(output.hasAttribute("batchAxis"));
            REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == 0);
            const auto outputId = output.id();

            Parser::Map rejected;
            rejected["url"] = std::string("ws://localhost:not-a-port/feed");
            REQUIRE(module->reconfigure(rejected) == Result::ERROR);
            REQUIRE(module->state() == Module::State::CREATED);
            const auto& retained =
                static_cast<const Modules::Websocket&>(module->config());
            REQUIRE(retained.url == config.url);
            REQUIRE(module->outputs().at("signal").tensor.id() == outputId);

            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
#endif
