#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy block delegates module configuration validation",
                 "[modules][io][soapy][block][validation]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    Parser::Map config;
    config["sampleRate"] = 0.0f;

    REQUIRE(flowgraph->blockCreate("soapy_bad_module_config", "soapy", config, {}) ==
            Result::SUCCESS);
    const auto block = viewBlock("soapy_bad_module_config");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE_FALSE(block.interfaceConfigs.empty());
    REQUIRE(block.diagnostic.find("[MODULE_SOAPY]") != std::string::npos);
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
    const auto block = viewBlock("soapy_bad_step");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.interfaceConfigs.empty());
    REQUIRE(block.outputs.empty());
}
