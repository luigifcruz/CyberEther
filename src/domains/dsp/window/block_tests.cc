#include <catch2/catch_test_macros.hpp>

#include <string>
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Window block creates with default config", "[modules][dsp][window][block]") {
    Blocks::Window config;

    REQUIRE(flowgraph->blockCreate("window_default", config, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("window_default")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("window_default")->outputs().at("window").tensor;
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE(out.rank() == 1);
    REQUIRE(out.shape(0) == 1024);
}

TEST_CASE_METHOD(FlowgraphFixture, "Window block creates with custom size", "[modules][dsp][window][block]") {
    Blocks::Window config;
    config.size = 256;

    REQUIRE(flowgraph->blockCreate("window_custom", config, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("window_custom")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("window_custom")->outputs().at("window").tensor;
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE(out.rank() == 1);
    REQUIRE(out.shape(0) == 256);
}

TEST_CASE_METHOD(FlowgraphFixture, "Window block rejects invalid size", "[modules][dsp][window][block][validation]") {
    Blocks::Window config;
    config.size = 0;

    REQUIRE(flowgraph->blockCreate("window_invalid", config, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("window_invalid")->state() == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture, "Window block reconfigure behavior", "[modules][dsp][window][block][reconfigure]") {
    Blocks::Window config;
    config.size = 64;

    REQUIRE(flowgraph->blockCreate("window_recfg", config, {}) == Result::SUCCESS);

    Parser::Map unchanged;
    unchanged["size"] = std::string("64");
    REQUIRE(flowgraph->blockReconfigure("window_recfg", unchanged) == Result::SUCCESS);

    const Tensor& unchangedOut = flowgraph->blockList().at("window_recfg")->outputs().at("window").tensor;
    REQUIRE(unchangedOut.shape(0) == 64);

    Parser::Map changed;
    changed["size"] = std::string("128");
    REQUIRE(flowgraph->blockReconfigure("window_recfg", changed) == Result::SUCCESS);

    const Tensor& outAfterReconfigure =
        flowgraph->blockList().at("window_recfg")->outputs().at("window").tensor;
    REQUIRE(outAfterReconfigure.shape(0) == 128);
    REQUIRE(flowgraph->blockList().at("window_recfg")->state() == Block::State::Created);
}
