#include <catch2/catch_test_macros.hpp>

#include <string>
#include "jetstream/domains/dsp/filter_taps/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Filter taps keeps legacy center-only head count", "[modules][dsp][filter_taps]") {
    Parser::Map config;
    config["sampleRate"] = std::string("2000000");
    config["bandwidth"] = std::string("200000");
    config["taps"] = std::string("51");
    config["center"] = std::string("[600000, 0]");

    REQUIRE(flowgraph->blockCreate("taps", "filter_taps", config, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("taps")->state() == Block::State::Created);

    const Tensor& coeffs = flowgraph->blockList().at("taps")->outputs().at("coeffs").tensor;
    REQUIRE(coeffs.rank() == 2);
    REQUIRE(coeffs.shape(0) == 2);
    REQUIRE(coeffs.shape(1) == 51);
}
