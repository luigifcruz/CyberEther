#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

#include "environment.hh"
#include "flowgraph_fixture.hh"
#include "jetstream/domains/io/soapy/block.hh"
#include "library/soapysdr.hh"
#include "scripted_soapy.hh"

using namespace Jetstream;

namespace {

struct RuntimeReset {
    RuntimeReset() {
        Modules::SoapyRuntime::Reset();
        Modules::SoapyDiscovery::ClearCache();
    }

    ~RuntimeReset() {
        Modules::SoapyRuntime::Reset();
        Modules::SoapyDiscovery::ClearCache();
    }
};

}  // namespace

TEST_CASE("Soapy block preserves public defaults and saved settings",
          "[modules][io][soapy][block][config]") {
    Blocks::Soapy config;
    REQUIRE(config.modulePath.empty());
    REQUIRE(config.deviceString.empty());
    REQUIRE(config.streamString.empty());
    REQUIRE(config.dataType == "CF32");
    REQUIRE(config.antenna.empty());
    REQUIRE(config.frequency == 96.9e6f);
    REQUIRE(config.frequencyStep == 1000000.0f);
    REQUIRE(config.sampleRate == 2.0e6f);
    REQUIRE(config.automaticGain);
    REQUIRE(config.manualGain == 0.0f);
    REQUIRE_FALSE(config.biasTee);
    REQUIRE(config.numberOfBatches == 8);
    REQUIRE(config.numberOfTimeSamples == 8192);
    REQUIRE(config.bufferMultiplier == 4);

    config.modulePath = "saved-device-module";
    config.deviceString = "driver=saved,serial=123";
    config.streamString = "bufflen=8192";
    config.dataType = "CI16";
    config.antenna = "RX";
    config.automaticGain = false;
    config.manualGain = 20.0f;
    config.biasTee = true;
    Parser::Map saved;
    REQUIRE(config.serialize(saved) == Result::SUCCESS);
    Blocks::Soapy restored;
    REQUIRE(restored.deserialize(saved) == Result::SUCCESS);
    REQUIRE(restored.hash() == config.hash());
    saved.erase("dataType");
    Blocks::Soapy legacy;
    REQUIRE(legacy.deserialize(saved) == Result::SUCCESS);
    REQUIRE(legacy.dataType == "CF32");
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy formats preserve old FM graphs and validate changed outputs",
                 "[modules][io][soapy][block][formats]") {
    const TestSoapy::Runtime runtime;
    TestSoapy::State().streamFormats = {"CF32", "CS16", "CS12"};
    Blocks::Soapy config;
    config.deviceString = "driver=scripted";
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    const auto radio = viewBlock("radio");
    REQUIRE(radio.state == Block::State::Created);
    const auto field = std::find_if(radio.interfaceConfigs.begin(),
                                    radio.interfaceConfigs.end(),
                                    [](const auto& entry) {
                                        return entry.name == "dataType";
                                    });
    REQUIRE(field != radio.interfaceConfigs.end());
    REQUIRE(Parser::Get<std::vector<Parser::Map>>(field->format, "options") ==
            std::vector<Parser::Map>{{{"label", "CF32"}, {"value", "CF32"}},
                                     {{"label", "CI16"}, {"value", "CI16"}}});
    TensorMap inputs;
    inputs["signal"].requested("radio", "signal");
    REQUIRE(flowgraph->blockCreate("fm", "fm", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("fm").state == Block::State::Created);
    REQUIRE(flowgraph->blockReconfigure("radio", {{"dataType", std::string("CI16")}}) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);
    REQUIRE(viewBlock("fm").state == Block::State::Errored);
    REQUIRE(flowgraph->blockReconfigure("radio", {{"dataType", std::string("CF32")}}) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("fm").state == Block::State::Created);
    REQUIRE(flowgraph->blockDestroy("fm", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("radio", false) == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy saved blocks stay recognizable without a system runtime",
                 "[modules][io][soapy][block][runtime]") {
    const TestSupport::ScopedEnvironment environment("CYBERETHER_SOAPYSDR_LIBRARY");
    REQUIRE(environment.set("/__cyberether_missing_soapy__/library"));
    const RuntimeReset reset;
    Blocks::Soapy config;
    config.modulePath = "saved-device-module";
    config.deviceString = "driver=saved,serial=123";
    config.streamString = "bufflen=8192";
    Parser::Map saved;
    REQUIRE(config.serialize(saved) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    const auto block = viewBlock("radio");
    REQUIRE(block.state == Block::State::Incomplete);
    REQUIRE(block.config == saved);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find("SoapySDR 0.8 was not found") != std::string::npos);
    REQUIRE(block.interfaceOutputs.size() == 1);
    REQUIRE(block.interfaceOutputs.front().name == "signal");
    const auto field = std::find_if(block.interfaceConfigs.begin(), block.interfaceConfigs.end(),
                                    [](const auto& entry) { return entry.name == "deviceString"; });
    REQUIRE(field != block.interfaceConfigs.end());
    const auto options = Parser::Get<std::vector<Parser::Map>>(field->format, "options");
    REQUIRE(options.back() == Parser::Map{{"label", "Configured device"},
                                         {"value", config.deviceString}});

    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("radio", metrics) == Result::SUCCESS);
    const std::vector<std::pair<std::string, Parser::Map>> formats{
        {"bufferFill", {{"type", "progressbar"}, {"warning_above", 0.5f}, {"error_above", 0.75f}}},
        {"bufferLoss", {{"type", "progressbar"}, {"warning_above", 0.0f}, {"error_above", 0.01f}}},
        {"throughput", {{"type", "progressbar"}, {"warning_below", 0.95f}, {"error_below", 0.8f}}},
    };
    for (const auto& [name, format] : formats) {
        INFO(name);
        const auto metric = std::find_if(metrics.begin(), metrics.end(),
                                         [name = name](const auto& entry) { return entry.name == name; });
        REQUIRE(metric != metrics.end());
        REQUIRE(metric->format == format);
    }
    const auto throughput = std::find_if(metrics.begin(), metrics.end(),
                                         [](const auto& entry) { return entry.name == "throughput"; });
    REQUIRE(std::isnan(std::any_cast<std::pair<std::string, F32>>(throughput->value).second));
    REQUIRE(flowgraph->blockDestroy("radio", false) == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy manual gain stays editable without a system runtime",
                 "[modules][io][soapy][block][gain]") {
    const TestSupport::ScopedEnvironment environment("CYBERETHER_SOAPYSDR_LIBRARY");
    REQUIRE(environment.set("/__cyberether_missing_soapy__/library"));
    const RuntimeReset reset;
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
                                      [](const auto& field) { return field.name == "antenna"; });
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
    REQUIRE(flowgraph->blockDestroy("radio", false) == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy blocks reject invalid block and module settings",
                 "[modules][io][soapy][block][validation]") {
    const TestSupport::ScopedEnvironment environment("CYBERETHER_SOAPYSDR_LIBRARY");
    REQUIRE(environment.set("/__cyberether_missing_soapy__/library"));
    const RuntimeReset reset;
    Blocks::Soapy config;
    std::string diagnostic;
    SECTION("block frequency step") {
        config.frequencyStep = 0.0f;
        diagnostic = "[BLOCK_SOAPY]";
    }
    SECTION("module sample rate") {
        config.sampleRate = 0.0f;
        diagnostic = "[MODULE_SOAPY]";
    }
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    const auto block = viewBlock("radio");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find(diagnostic) != std::string::npos);
    REQUIRE(flowgraph->blockDestroy("radio", false) == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy controls follow capabilities immediately and after device changes",
                 "[modules][io][soapy][block][capabilities]") {
    const TestSoapy::Runtime runtime;
    TestSoapy::State().automaticGainSupported = false;
    Blocks::Soapy config;
    config.deviceString = "driver=scripted";
    config.manualGain = 18.0f;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    const auto hasField = [&](const std::string& name) {
        const auto block = viewBlock("radio");
        return std::any_of(block.interfaceConfigs.begin(), block.interfaceConfigs.end(),
                           [&](const auto& field) { return field.name == name; });
    };
    REQUIRE_FALSE(hasField("automaticGain"));
    REQUIRE_FALSE(hasField("antenna"));
    REQUIRE_FALSE(hasField("biasTee"));
    REQUIRE(hasField("manualGain"));
    REQUIRE(TestSoapy::State().settings().gain == 18.0);
    REQUIRE(Parser::Get<bool>(viewBlock("radio").config, "automaticGain"));

    TestSoapy::State().automaticGainSupported = true;
    REQUIRE(flowgraph->blockReconfigure("radio", {{"deviceString", "driver=scripted,new=1"}}) ==
            Result::SUCCESS);
    REQUIRE(hasField("automaticGain"));
    REQUIRE_FALSE(hasField("manualGain"));
    REQUIRE(flowgraph->blockReconfigure("radio", {{"automaticGain", false}}) ==
            Result::SUCCESS);
    REQUIRE(hasField("manualGain"));
    REQUIRE(flowgraph->blockReconfigure("radio", {{"automaticGain", true}}) ==
            Result::SUCCESS);
    REQUIRE_FALSE(hasField("manualGain"));
    REQUIRE(flowgraph->blockDestroy("radio", false) == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy actual settings show rounded values without changing saved requests",
                 "[modules][io][soapy][block][readback]") {
    const TestSoapy::Runtime runtime;
    TestSoapy::State().sampleRateStep = 1.0e6;
    Blocks::Soapy config;
    config.deviceString = "driver=scripted";
    config.sampleRate = 2.4e6f;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    const auto queries = TestSoapy::State().queries.load();
    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("radio", metrics) == Result::SUCCESS);
    const auto metric = std::find_if(metrics.begin(), metrics.end(), [](const auto& entry) {
        return entry.name == "actualSettings";
    });
    REQUIRE(metric != metrics.end());
    const auto table = std::any_cast<Parser::Map>(metric->value);
    const auto rows = Parser::Get<std::vector<std::vector<std::string>>>(table, "rows");
    REQUIRE(std::find(rows.begin(), rows.end(),
                      std::vector<std::string>{"Sample Rate", "2000000 S/s"}) != rows.end());
    REQUIRE(Parser::Get<F32>(viewBlock("radio").config, "sampleRate") == 2.4e6f);
    REQUIRE(TestSoapy::State().queries == queries);
    REQUIRE(flowgraph->blockDestroy("radio", false) == Result::SUCCESS);
}
