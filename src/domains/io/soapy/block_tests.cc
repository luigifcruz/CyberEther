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
    REQUIRE(viewBlock("soapy_invalid").state ==
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
    REQUIRE(viewBlock("soapy_bad_samples").state ==
            Block::State::Errored);

    Parser::Map badMultiplier;
    badMultiplier["bufferMultiplier"] = std::string("0");

    REQUIRE(flowgraph->blockCreate("soapy_bad_multiplier", "soapy",
                                   badMultiplier, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("soapy_bad_multiplier").state ==
            Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy block rejects invalid frequency steps",
                 "[modules][io][soapy][block][validation]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    Parser::Map config;
    config["frequencyStep"] = 0.0f;

    REQUIRE(flowgraph->blockCreate("soapy_bad_step", "soapy", config, {}) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("soapy_bad_step").state == Block::State::Errored);
}
