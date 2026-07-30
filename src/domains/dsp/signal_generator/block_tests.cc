#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal generator block creates with default config",
                 "[modules][dsp][signal_generator][block]") {
    REQUIRE(flowgraph->blockCreate("gen", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("gen").state == Block::State::Created);

    const Tensor out = viewBlock("gen").outputs.at("signal").tensor;
    REQUIRE(out.rank() == 1);
    REQUIRE(out.shape(0) == 8192);
    REQUIRE(out.dtype() == DataType::F32);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal generator block recreates on signal type change",
                 "[modules][dsp][signal_generator][block][reconfigure]") {
    REQUIRE(flowgraph->blockCreate("gen", "signal_generator", {}, {}) == Result::SUCCESS);

    Parser::Map update;
    update["signalType"] = std::string("chirp");

    REQUIRE(flowgraph->blockReconfigure("gen", update) == Result::SUCCESS);
    REQUIRE(viewBlock("gen").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal generator consecutive sparse updates preserve each other",
                 "[modules][dsp][signal_generator][block][reconfigure][sparse]") {
    Blocks::SignalGenerator config;
    config.frequency = 2000.0f;
    config.amplitude = 0.25f;
    config.phase = 0.5f;
    REQUIRE(flowgraph->blockCreate("gen", config, {}) == Result::SUCCESS);

    Parser::Map frequencyUpdate;
    frequencyUpdate["frequency"] = 3000.0f;
    REQUIRE(flowgraph->blockReconfigure("gen", frequencyUpdate) == Result::SUCCESS);

    Parser::Map amplitudeUpdate;
    amplitudeUpdate["amplitude"] = 0.75f;
    REQUIRE(flowgraph->blockReconfigure("gen", amplitudeUpdate) == Result::SUCCESS);

    Parser::Map savedMap;
    REQUIRE(flowgraph->blockConfig("gen", savedMap) == Result::SUCCESS);
    Blocks::SignalGenerator saved;
    REQUIRE(saved.deserialize(savedMap) == Result::SUCCESS);
    REQUIRE(saved.frequency == 3000.0f);
    REQUIRE(saved.amplitude == 0.75f);
    REQUIRE(saved.phase == 0.5f);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Signal generator preserves invalid updates for recovery",
                 "[modules][dsp][signal_generator][block][reconfigure][validation]") {
    Blocks::SignalGenerator config;
    REQUIRE(flowgraph->blockCreate("gen", config, {}) == Result::SUCCESS);
    Parser::Map invalidUpdate;
    invalidUpdate["sampleRate"] = std::numeric_limits<F32>::quiet_NaN();
    REQUIRE(flowgraph->blockReconfigure("gen", invalidUpdate) == Result::SUCCESS);
    REQUIRE(viewBlock("gen").state == Block::State::Errored);
    REQUIRE(viewBlock("gen").outputs.empty());

    Parser::Map savedMap;
    REQUIRE(flowgraph->blockConfig("gen", savedMap) == Result::SUCCESS);
    Blocks::SignalGenerator saved;
    REQUIRE(saved.deserialize(savedMap) == Result::SUCCESS);
    REQUIRE(std::isnan(saved.sampleRate));

    Parser::Map recovery;
    recovery["sampleRate"] = config.sampleRate;
    REQUIRE(flowgraph->blockReconfigure("gen", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("gen").state == Block::State::Created);
    REQUIRE(viewBlock("gen").outputs.contains("signal"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal generator exposes candidate chirp fields before validation fails",
                 "[modules][dsp][signal_generator][block][validation][interface]") {
    Blocks::SignalGenerator config;
    config.signalType = "chirp";
    config.sampleRate = std::numeric_limits<F32>::quiet_NaN();
    REQUIRE(flowgraph->blockCreate("chirp_bad", config, {}) == Result::SUCCESS);

    const auto block = viewBlock("chirp_bad");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(std::any_of(block.interfaceConfigs.begin(),
                        block.interfaceConfigs.end(),
                        [](const auto& field) { return field.name == "chirpStartFreq"; }));
    REQUIRE(std::any_of(block.interfaceConfigs.begin(),
                        block.interfaceConfigs.end(),
                        [](const auto& field) { return field.name == "chirpEndFreq"; }));
    REQUIRE(std::any_of(block.interfaceConfigs.begin(),
                        block.interfaceConfigs.end(),
                        [](const auto& field) { return field.name == "chirpDuration"; }));
}
