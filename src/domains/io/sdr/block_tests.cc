#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/io/sdr/block.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("SDR block keeps the radio interface defaults",
          "[modules][io][sdr][block]") {
    Blocks::Sdr config;
    REQUIRE(config.deviceString.empty());
    REQUIRE(config.antenna.empty());
    REQUIRE(config.frequency == 96.9e6f);
    REQUIRE(config.frequencyStep == 1000000.0f);
    REQUIRE(config.sampleRate == 2.0e6f);
    REQUIRE(config.automaticGain);
    REQUIRE_FALSE(config.biasTee);
    REQUIRE(config.numberOfBatches == 8);
    REQUIRE(config.numberOfTimeSamples == 8192);
    REQUIRE(config.bufferMultiplier == 4);
}

TEST_CASE("SDR block selections and settings round-trip through saved configuration",
          "[modules][io][sdr][block]") {
    Blocks::Sdr config;
    config.deviceString = "driver=hackrf,serial=0000000000000000a06063c8234e925f";
    config.antenna = "LNAW";
    config.biasTee = true;
    config.frequency = 433.92e6f;

    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE_FALSE(serialized.contains("modulePath"));
    REQUIRE_FALSE(serialized.contains("streamString"));

    Blocks::Sdr restored;
    REQUIRE(restored.deserialize(serialized) == Result::SUCCESS);
    REQUIRE(restored.deviceString == config.deviceString);
    REQUIRE(restored.antenna == config.antenna);
    REQUIRE(restored.biasTee);
    REQUIRE(restored.frequency == config.frequency);
    REQUIRE(restored.hash() == config.hash());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "SDR block exposes the radio controls and graphical metrics",
                 "[modules][io][sdr][block]") {
    REQUIRE_FALSE(Registry::ListAvailableModules("sdr").empty());

    Parser::Map config;
    config["frequencyStep"] = 0.0f;
    REQUIRE(flowgraph->blockCreate("bad_step", "sdr", config, {}) == Result::SUCCESS);
    const auto bad = viewBlock("bad_step");
    REQUIRE(bad.state == Block::State::Errored);
    REQUIRE(bad.diagnostic.find("[BLOCK_SDR]") != std::string::npos);

    REQUIRE(flowgraph->blockCreate("radio", "sdr", {}, {}) == Result::SUCCESS);
    const auto block = viewBlock("radio");
    REQUIRE(block.state == Block::State::Incomplete);
    REQUIRE(block.interfaceOutputs.size() == 1);
    REQUIRE(block.interfaceOutputs.front().name == "signal");
    for (const auto* name : {"deviceString", "antenna", "frequency", "sampleRate",
                             "automaticGain", "biasTee", "numberOfBatches",
                             "numberOfTimeSamples", "bufferMultiplier"}) {
        INFO(name);
        REQUIRE(std::any_of(block.interfaceConfigs.begin(), block.interfaceConfigs.end(),
                            [&](const auto& field) { return field.name == name; }));
    }

    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("radio", metrics) == Result::SUCCESS);
    for (const auto* name : {"bufferHealth", "bufferLoss", "throughput"}) {
        INFO(name);
        const auto metric = std::find_if(metrics.begin(), metrics.end(),
                                         [&](const auto& entry) { return entry.name == name; });
        REQUIRE(metric != metrics.end());
        REQUIRE(metric->format == Parser::Map{{"type", "progressbar"}});
    }
}
