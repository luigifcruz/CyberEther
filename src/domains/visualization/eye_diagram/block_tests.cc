#include <catch2/catch_test_macros.hpp>

#include <jetstream/domains/dsp/signal_generator/block.hh>
#include <jetstream/domains/visualization/eye_diagram/block.hh>
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Eye Diagram block wiring and control changes",
                 "[modules][eye_diagram][block]") {
    Blocks::SignalGenerator source;
    source.signalDataType = "CF32";
    source.bufferSize = 128;
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
    TensorMap inputs;
    inputs["signal"].requested("src", "signal");
    REQUIRE(flowgraph->blockCreate("eye", Blocks::EyeDiagram{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("eye").state == Block::State::Created);
    REQUIRE(viewBlock("eye").outputs.empty());
    REQUIRE(flowgraph->blockReconfigure("eye", {
        {"samplesPerSymbol", U64{4}}, {"interpolation", U64{8}},
        {"component", std::string("imag")},
        {"rangeMin", -3.0f}, {"rangeMax", 3.0f}
    }) == Result::SUCCESS);
    REQUIRE(viewBlock("eye").state == Block::State::Created);
    auto result = flowgraph->blockDisconnect("eye", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("eye").state == Block::State::Incomplete);
    REQUIRE(flowgraph->blockConnect("eye", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("eye").state == Block::State::Created);
}
