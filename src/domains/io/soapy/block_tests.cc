#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/io/soapy/block.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("Soapy block Bias-T defaults off", "[modules][io][soapy][block][bias-tee]") {
    Blocks::Soapy config;
    REQUIRE_FALSE(config.biasTee);

    config.biasTee = true;
    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);

    Blocks::Soapy restored;
    REQUIRE(restored.deserialize(serialized) == Result::SUCCESS);
    REQUIRE(restored.biasTee);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy block delegates module configuration validation",
                 "[modules][io][soapy][block][validation]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    Parser::Map config;
    config["deviceString"] = std::string("driver=cyberether_missing_test_driver");
    config["sampleRate"] = 0.0f;

    REQUIRE(flowgraph->blockCreate("soapy_bad_module_config", "soapy", config, {}) ==
            Result::SUCCESS);
    const auto block = viewBlock("soapy_bad_module_config");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE_FALSE(block.interfaceConfigs.empty());
    REQUIRE(block.diagnostic.find("[MODULE_SOAPY]") != std::string::npos);

    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("soapy_bad_module_config", metrics) == Result::SUCCESS);
    const auto loss = std::find_if(metrics.begin(), metrics.end(), [](const auto& metric) {
        return metric.name == "bufferLoss";
    });
    REQUIRE(loss != metrics.end());
    REQUIRE(loss->label == "Buffer Loss");
    REQUIRE(loss->format == Parser::Map{{"type", "progressbar"}});
    const auto [label, fraction] = std::any_cast<std::pair<std::string, F32>>(loss->value);
    REQUIRE(label == "0.00%");
    REQUIRE(fraction == 0.0f);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy block rejects invalid frequency steps",
                 "[modules][io][soapy][block][validation]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    Parser::Map config;
    config["deviceString"] = std::string("driver=cyberether_missing_test_driver");
    config["frequencyStep"] = 0.0f;

    REQUIRE(flowgraph->blockCreate("soapy_bad_step", "soapy", config, {}) ==
            Result::SUCCESS);
    const auto block = viewBlock("soapy_bad_step");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE_FALSE(block.interfaceOutputs.empty());
    REQUIRE_FALSE(block.interfaceConfigs.empty());
    REQUIRE(std::none_of(block.interfaceConfigs.begin(),
                         block.interfaceConfigs.end(),
                         [](const auto& field) { return field.name == "modulePath"; }));
    REQUIRE(block.outputs.empty());
    const auto biasTee = std::find_if(block.interfaceConfigs.begin(),
                                      block.interfaceConfigs.end(),
                                      [](const auto& field) {
                                          return field.name == "biasTee";
                                      });
    REQUIRE(biasTee != block.interfaceConfigs.end());
    REQUIRE(biasTee->label == "Bias-T");
    REQUIRE(biasTee->format == Parser::Map{{"type", "bool"}});
}
