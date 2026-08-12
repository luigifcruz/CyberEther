#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/dsp/psk_demod/block.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "PSK demod block creates and exposes output",
                 "[modules][dsp][psk_demod][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("1024");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("demod", "psk_demod", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("demod").state == Block::State::Created);

    const Tensor out = viewBlock("demod").outputs.at("signal").tensor;
    REQUIRE(out.rank() == 1);
    REQUIRE(out.dtype() == DataType::CF32);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "PSK demod block delegates dtype validation to its module",
                 "[modules][dsp][psk_demod][block][validation]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("F32");
    sourceConfig["bufferSize"] = std::string("1024");

    REQUIRE(flowgraph->blockCreate("psk_dtype_src", "signal_generator",
                                   sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("psk_dtype_src", "signal");

    REQUIRE(flowgraph->blockCreate("psk_dtype_bad", "psk_demod", {}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("psk_dtype_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("psk_dtype_bad").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "PSK demod block preserves invalid symbol rate for recovery",
                 "[modules][dsp][psk_demod][block][reconfigure][validation]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("1024");

    REQUIRE(flowgraph->blockCreate("psk_update_src", "signal_generator",
                                   sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("psk_update_src", "signal");

    Blocks::PskDemod config;
    REQUIRE(flowgraph->blockCreate("psk_update", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const auto outputId = viewBlock("psk_update").outputs.at("signal").tensor.id();

    Parser::Map loopUpdate;
    loopUpdate["frequencyLoopBandwidth"] = F32{0.1f};
    loopUpdate["timingLoopBandwidth"] = F32{0.1f};
    loopUpdate["dampingFactor"] = F32{1.0f};
    REQUIRE(flowgraph->blockReconfigure("psk_update", loopUpdate) == Result::SUCCESS);
    REQUIRE(viewBlock("psk_update").outputs.at("signal").tensor.id() == outputId);

    Parser::Map invalidUpdate;
    invalidUpdate["symbolRate"] = F32{1500000.0f};
    REQUIRE(flowgraph->blockReconfigure("psk_update", invalidUpdate) == Result::SUCCESS);
    REQUIRE(viewBlock("psk_update").state == Block::State::Errored);
    REQUIRE(viewBlock("psk_update").outputs.empty());

    Parser::Map savedMap;
    REQUIRE(flowgraph->blockConfig("psk_update", savedMap) == Result::SUCCESS);
    Blocks::PskDemod saved;
    REQUIRE(saved.deserialize(savedMap) == Result::SUCCESS);
    REQUIRE(saved.symbolRate == 1500000.0f);
    REQUIRE(saved.frequencyLoopBandwidth == 0.1f);
    REQUIRE(saved.timingLoopBandwidth == 0.1f);
    REQUIRE(saved.dampingFactor == 1.0f);

    Parser::Map recovery;
    recovery["symbolRate"] = config.symbolRate;
    REQUIRE(flowgraph->blockReconfigure("psk_update", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("psk_update").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}
