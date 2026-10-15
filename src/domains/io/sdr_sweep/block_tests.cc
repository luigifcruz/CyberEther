#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/io/sdr_sweep/block.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("SDR Sweep block defaults to a manual-gain sweep",
          "[modules][io][sdr_sweep][block]") {
    Blocks::SdrSweep config;
    REQUIRE(config.deviceString.empty());
    REQUIRE_FALSE(config.automaticGain);
    REQUIRE(config.startFrequency == 88.0e6);
    REQUIRE(config.endFrequency == 108.0e6);
    REQUIRE(config.sampleRate == 0.0);
    REQUIRE(config.resolution == 10.0e3);
    REQUIRE(config.dwell == 1);
    REQUIRE(config.settlingTime == 0.0f);
    config.startFrequency = 2400.0e6;
    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE_FALSE(serialized.contains("sweepMethod"));
    serialized["sweepMethod"] = "bladerf_quick_tune";
    Blocks::SdrSweep restored;
    REQUIRE(restored.deserialize(serialized) == Result::SUCCESS);
    REQUIRE(restored.startFrequency == config.startFrequency);
    Parser::Map saved;
    REQUIRE(restored.serialize(saved) == Result::SUCCESS);
    REQUIRE_FALSE(saved.contains("sweepMethod"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "SDR Sweep block offers buffer bars and text sweep speed",
                 "[modules][io][sdr_sweep][block]") {
    REQUIRE_FALSE(Registry::ListAvailableModules("sdr_sweep").empty());
    REQUIRE(flowgraph->blockCreate("sweep", "sdr_sweep", {}, {}) == Result::SUCCESS);
    const auto block = viewBlock("sweep");
    REQUIRE(block.state == Block::State::Incomplete);

    REQUIRE(std::none_of(block.interfaceConfigs.begin(), block.interfaceConfigs.end(),
                         [](const auto& field) { return field.name == "sweepMethod" || field.label == "Method"; }));

    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("sweep", metrics) == Result::SUCCESS);
    REQUIRE(metrics.size() == 3);
    const std::vector<std::string> names{"bufferHealth", "bufferLoss", "sweepSpeed"};
    const std::vector<std::string> labels{"Buffer Health", "Buffer Loss", "Sweep Speed"};
    const std::vector<std::string> expectedValues{"0.0%", "0.00%", "0.00 GHz/s (0.0 Hz)"};
    for (U64 i = 0; i < metrics.size(); ++i) {
        REQUIRE(metrics[i].name == names[i]);
        REQUIRE(metrics[i].label == labels[i]);
    }
    for (U64 i = 0; i < 2; ++i) {
        REQUIRE(metrics[i].format == Parser::Map{{"type", "progressbar"}});
        const auto [value, fraction] = std::any_cast<std::pair<std::string, F32>>(metrics[i].value);
        REQUIRE(value == expectedValues[i]);
        REQUIRE(fraction == 0.0f);
    }
    REQUIRE(metrics[2].format == Parser::Map{{"type", "label"}});
    REQUIRE(std::any_cast<std::string>(metrics[2].value) == expectedValues[2]);
    REQUIRE(metrics[0].help.find("software slot") != std::string::npos);
    REQUIRE(metrics[1].help.find("completed frames produced") != std::string::npos);
    REQUIRE(metrics[2].help.find("complete sweeps per second") != std::string::npos);
}
