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

TEST_CASE("Soapy antenna selection is optional and serialized", "[modules][io][soapy][block][antenna]") {
    Blocks::Soapy config;
    REQUIRE(config.deserialize({{"deviceString", "driver=test"}}) == Result::SUCCESS);
    REQUIRE(config.antenna.empty());
    const auto defaultHash = config.hash();
    config.antenna = "TX/RX";
    REQUIRE(config.hash() != defaultHash);
    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    Blocks::Soapy restored;
    REQUIRE(restored.deserialize(serialized) == Result::SUCCESS);
    REQUIRE(restored.antenna == config.antenna);
    REQUIRE(restored.hash() == config.hash());
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
    REQUIRE(std::none_of(metrics.begin(), metrics.end(), [](const auto& metric) {
        return metric.name == "deviceOverflows";
    }));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy frequency step remains editable without a selected device",
                 "[modules][io][soapy][block][antenna][reconfigure]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    Blocks::Soapy config;
    config.antenna = "RX";
    config.automaticGain = false;
    config.manualGain = 20.0f;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockReconfigure("radio", {{"frequencyStep", 2000000.0f}}) ==
            Result::SUCCESS);
    const auto block = viewBlock("radio");
    REQUIRE(block.state == Block::State::Incomplete);
    REQUIRE(Parser::Get<F32>(block.config, "frequencyStep") == 2000000.0f);
    REQUIRE(block.outputs.empty());
    const auto antenna = std::find_if(block.interfaceConfigs.begin(),
                                      block.interfaceConfigs.end(),
                                      [](const auto& field) {
                                          return field.name == "antenna";
                                      });
    REQUIRE(antenna != block.interfaceConfigs.end());
    REQUIRE(Parser::Get<std::vector<Parser::Map>>(antenna->format, "options") ==
            std::vector<Parser::Map>{{{"label", "Default"}, {"value", ""}},
                                    {{"label", "RX"}, {"value", "RX"}}});
    const auto gain = std::find_if(block.interfaceConfigs.begin(),
                                  block.interfaceConfigs.end(),
                                  [](const auto& field) { return field.name == "manualGain"; });
    REQUIRE(gain != block.interfaceConfigs.end());
    REQUIRE(gain->format == Parser::Map{
        {"type", "range"}, {"min", 0.0f}, {"max", 60.0f}, {"unit", "dB"},
    });
    REQUIRE(Parser::Get<F32>(block.config, "manualGain") == 20.0f);
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
