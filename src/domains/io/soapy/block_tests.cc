#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy block rejects invalid batch dimensions",
                 "[modules][io][soapy][block][validation]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    Parser::Map config;
    config["numberOfBatches"] = std::string("0");

    REQUIRE(flowgraph->blockCreate("soapy_invalid", "soapy", config, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("soapy_invalid")->state() ==
            Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy block rejects invalid sample dimensions",
                 "[modules][io][soapy][block][validation]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    Parser::Map badSamples;
    badSamples["numberOfTimeSamples"] = std::string("0");

    REQUIRE(flowgraph->blockCreate("soapy_bad_samples", "soapy", badSamples,
                                   {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("soapy_bad_samples")->state() ==
            Block::State::Errored);

    Parser::Map badMultiplier;
    badMultiplier["bufferMultiplier"] = std::string("0");

    REQUIRE(flowgraph->blockCreate("soapy_bad_multiplier", "soapy",
                                   badMultiplier, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("soapy_bad_multiplier")->state() ==
            Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy block reconfigure keeps output shape",
                 "[modules][io][soapy][block][reconfigure]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    Parser::Map invalidStart;
    invalidStart["numberOfBatches"] = std::string("0");
    REQUIRE(flowgraph->blockCreate("soapy_cfg", "soapy", invalidStart, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("soapy_cfg")->state() ==
            Block::State::Errored);

    Parser::Map update;
    update["hintString"] = std::string("driver=mock");
    update["deviceString"] = std::string("driver=mock");
    update["streamString"] = std::string("bufflen=8192");
    update["frequency"] = std::string("100500000");
    update["sampleRate"] = std::string("1500000");
    update["automaticGain"] = std::string("false");
    update["numberOfBatches"] = std::string("4");
    update["numberOfTimeSamples"] = std::string("256");
    update["bufferMultiplier"] = std::string("2");
    REQUIRE(flowgraph->blockReconfigure("soapy_cfg", update) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().find("soapy_cfg") != flowgraph->blockList().end());
}
