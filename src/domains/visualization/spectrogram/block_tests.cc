#include <catch2/catch_test_macros.hpp>

#include <string>

#include "jetstream/domains/core/ones_tensor/block.hh"
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

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrogram block delegates rank validation to its module",
                 "[modules][spectrogram][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {2, 2, 2};
    source.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("spectrogram_rank_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("spectrogram_rank_src", "buffer");

    REQUIRE(flowgraph->blockCreate("spectrogram_rank", Blocks::Spectrogram{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("spectrogram_rank").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrogram block delegates dtype validation to its module",
                 "[modules][spectrogram][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("spectrogram_dtype_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("spectrogram_dtype_src", "buffer");

    REQUIRE(flowgraph->blockCreate("spectrogram_dtype",
                                   Blocks::Spectrogram{}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("spectrogram_dtype").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Spectrogram block preserves invalid height for recovery",
                 "[modules][spectrogram][block][reconfigure][validation]") {
    Blocks::SignalGenerator source;
    source.signalDataType = "F32";
    source.bufferSize = 64;
    REQUIRE(flowgraph->blockCreate("spectrogram_update_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("spectrogram_update_src", "signal");

    Blocks::Spectrogram config;
    config.height = 128;
    REQUIRE(flowgraph->blockCreate("spectrogram_update", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["height"] = U64{0};
    REQUIRE(flowgraph->blockReconfigure("spectrogram_update", update) == Result::SUCCESS);
    REQUIRE(viewBlock("spectrogram_update").state == Block::State::Errored);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("spectrogram_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("height")) == 0);

    Parser::Map recovery;
    recovery["height"] = config.height;
    REQUIRE(flowgraph->blockReconfigure("spectrogram_update", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("spectrogram_update").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}
