#include <catch2/catch_test_macros.hpp>

#include <string>

#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/visualization/eye_diagram/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Eye Diagram block wiring and lifecycle",
                 "[modules][eye_diagram][block]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "CF32";
    sourceConfig.bufferSize = 128;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("eye", Blocks::EyeDiagram{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("eye").state == Block::State::Created);
    REQUIRE(viewBlock("eye").outputs.empty());

    Parser::Map update;
    update["samplesPerSymbol"] = U64{4};
    update["interpolation"] = U64{8};
    update["component"] = std::string("imag");
    update["rangeMin"] = -3.0f;
    update["rangeMax"] = 3.0f;
    REQUIRE(flowgraph->blockReconfigure("eye", update) == Result::SUCCESS);
    REQUIRE(viewBlock("eye").state == Block::State::Created);

    auto result = flowgraph->blockDisconnect("eye", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("eye").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("eye", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("eye").state == Block::State::Created);
}
