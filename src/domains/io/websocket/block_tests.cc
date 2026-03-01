#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/registry.hh"

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
    REQUIRE(flowgraph->blockList().at("ws_invalid")->state() ==
            Block::State::Errored);
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
    REQUIRE(flowgraph->blockList().at("ws_bad_batches")->state() ==
            Block::State::Errored);

    Parser::Map zeroSamples;
    zeroSamples["numberOfTimeSamples"] = std::string("0");
    REQUIRE(flowgraph->blockCreate("ws_bad_samples", "websocket",
                                   zeroSamples, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("ws_bad_samples")->state() ==
            Block::State::Errored);

    Parser::Map zeroMultiplier;
    zeroMultiplier["bufferMultiplier"] = std::string("0");
    REQUIRE(flowgraph->blockCreate("ws_bad_multiplier", "websocket",
                                   zeroMultiplier, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("ws_bad_multiplier")->state() ==
            Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Websocket block reconfigure updates output dimensions",
                 "[modules][io][websocket][block][reconfigure]") {
    if (Registry::ListAvailableModules("websocket").empty()) {
        SUCCEED("Websocket module is unavailable in this build.");
        return;
    }

    Parser::Map config;
    config["url"] = std::string("");
    REQUIRE(flowgraph->blockCreate("ws_cfg", "websocket", config, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("ws_cfg")->state() ==
            Block::State::Incomplete);

    Parser::Map reconfigure;
    reconfigure["url"] = std::string("ws://localhost:9000");
    reconfigure["dataType"] = std::string("CU8");
    reconfigure["numberOfBatches"] = std::string("2");
    reconfigure["numberOfTimeSamples"] = std::string("256");
    reconfigure["bufferMultiplier"] = std::string("2");
    REQUIRE(flowgraph->blockReconfigure("ws_cfg", reconfigure) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().find("ws_cfg") != flowgraph->blockList().end());
}
