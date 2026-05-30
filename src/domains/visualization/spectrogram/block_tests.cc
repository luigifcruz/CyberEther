#include <catch2/catch_test_macros.hpp>

#include <string>

#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/visualization/spectrogram/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrogram block create and lifecycle",
                 "[modules][spectrogram][block]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    Blocks::Spectrogram config;
    config.height = 128;

    REQUIRE(flowgraph->blockCreate("spectrogram", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("spectrogram").state ==
            Block::State::Created);
    REQUIRE(viewBlock("spectrogram").outputs.empty());

    auto result = flowgraph->blockDisconnect("spectrogram", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("spectrogram").state ==
            Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("spectrogram", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("spectrogram").state ==
            Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrogram block reconfigure and validation",
                 "[modules][spectrogram][block][validation]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("spectrogram", Blocks::Spectrogram(), inputs) ==
            Result::SUCCESS);

    Parser::Map config;
    config["height"] = std::string("64");
    REQUIRE(flowgraph->blockReconfigure("spectrogram", config) == Result::SUCCESS);
    REQUIRE(viewBlock("spectrogram").state ==
            Block::State::Created);

    Blocks::Spectrogram invalid;
    invalid.height = 0;
    REQUIRE(flowgraph->blockCreate("spectrogram_invalid", invalid, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("spectrogram_invalid").state ==
            Block::State::Errored);
}
