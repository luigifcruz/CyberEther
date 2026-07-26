#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/registry.hh"
#include "test_server.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Websocket block rejects invalid data type",
                 "[modules][io][websocket][block][validation]") {
    if (Registry::ListAvailableModules("websocket").empty()) {
        SUCCEED("Websocket module is unavailable in this build.");
        return;
    }

    Parser::Map config;
    config["dataType"] = std::string("I32");

    REQUIRE(flowgraph->blockCreate("ws_invalid", "websocket", config, {}) ==
            Result::SUCCESS);
    const auto block = viewBlock("ws_invalid");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find("[MODULE_WEBSOCKET]") != std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Websocket block propagates module URL validation",
                 "[modules][io][websocket][block][validation]") {
    if (Registry::ListAvailableModules("websocket").empty()) {
        SUCCEED("Websocket module is unavailable in this build.");
        return;
    }

    Parser::Map config;
    config["url"] = std::string("ws://localhost:9000/feed#fragment");
    REQUIRE(flowgraph->blockCreate("ws_bad_url", "websocket", config, {}) ==
            Result::SUCCESS);

    const auto block = viewBlock("ws_bad_url");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find("[MODULE_WEBSOCKET]") != std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Websocket block rejects invalid dimensions",
                 "[modules][io][websocket][block][validation]") {
    if (Registry::ListAvailableModules("websocket").empty()) {
        SUCCEED("Websocket module is unavailable in this build.");
        return;
    }

    Parser::Map zeroBatches;
    zeroBatches["numberOfBatches"] = std::string("0");
    REQUIRE(flowgraph->blockCreate("ws_bad_batches", "websocket",
                                   zeroBatches, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("ws_bad_batches").state ==
            Block::State::Errored);

    Parser::Map zeroSamples;
    zeroSamples["numberOfTimeSamples"] = std::string("0");
    REQUIRE(flowgraph->blockCreate("ws_bad_samples", "websocket",
                                   zeroSamples, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("ws_bad_samples").state ==
            Block::State::Errored);

    Parser::Map zeroMultiplier;
    zeroMultiplier["bufferMultiplier"] = std::string("0");
    REQUIRE(flowgraph->blockCreate("ws_bad_multiplier", "websocket",
                                   zeroMultiplier, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("ws_bad_multiplier").state ==
            Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Websocket block recreates from incomplete with a valid URL",
                 "[modules][io][websocket][block][reconfigure]") {
#ifdef JST_OS_BROWSER
    SUCCEED("Native lifecycle coverage requires the local WebSocket server.");
    return;
#else
    if (Registry::ListAvailableModules("websocket").empty()) {
        SUCCEED("Websocket module is unavailable in this build.");
        return;
    }

    Tests::WebsocketTestServer server;
    REQUIRE(server.valid());

    Parser::Map config;
    config["url"] = std::string("");
    REQUIRE(flowgraph->blockCreate("ws_lifecycle", "websocket", config, {}) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("ws_lifecycle").state ==
            Block::State::Incomplete);

    Parser::Map reconfigure;
    reconfigure["url"] = server.url();
    reconfigure["dataType"] = std::string("U8");
    reconfigure["numberOfBatches"] = std::string("1");
    reconfigure["numberOfTimeSamples"] = std::string("1");
    reconfigure["bufferMultiplier"] = std::string("1");
    REQUIRE(flowgraph->blockReconfigure("ws_lifecycle", reconfigure) ==
            Result::SUCCESS);

    const auto block = viewBlock("ws_lifecycle");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("signal").tensor.dtype() == DataType::U8);
    REQUIRE(block.outputs.at("signal").tensor.shape() == Shape{1, 1});
    REQUIRE(flowgraph->blockDestroy("ws_lifecycle", false) == Result::SUCCESS);
#endif
}
