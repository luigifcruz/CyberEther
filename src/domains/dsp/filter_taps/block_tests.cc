#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>
#include <vector>

#include "jetstream/domains/dsp/filter_taps/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Filter taps uses heads over center length", "[modules][dsp][filter_taps]") {
    Parser::Map config;
    config["sampleRate"] = std::string("2000000");
    config["bandwidth"] = std::string("200000");
    config["taps"] = std::string("51");
    config["heads"] = std::string("1");
    config["center"] = std::string("[600000, 0]");

    REQUIRE(flowgraph->blockCreate("taps", "filter_taps", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("taps").state == Block::State::Created);

    const Tensor coeffs = viewBlock("taps").outputs.at("coeffs").tensor;
    REQUIRE(coeffs.rank() == 2);
    REQUIRE(coeffs.shape(0) == 1);
    REQUIRE(coeffs.shape(1) == 51);
}

TEST_CASE_METHOD(FlowgraphFixture, "Filter taps heads can shrink stale center vector", "[modules][dsp][filter_taps][reconfigure]") {
    Blocks::FilterTaps config;
    config.taps = 51;
    config.heads = 5;
    config.center = {0.0f, 0.1e6f, -0.1e6f, 0.2e6f, -0.2e6f};

    REQUIRE(flowgraph->blockCreate("taps_shrink", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("taps_shrink").state == Block::State::Created);

    Parser::Map update;
    update["sampleRate"] = config.sampleRate;
    update["bandwidth"] = config.bandwidth;
    update["taps"] = config.taps;
    update["heads"] = static_cast<U64>(1);
    update["center"] = config.center;

    REQUIRE(flowgraph->blockReconfigure("taps_shrink", update) == Result::SUCCESS);
    REQUIRE(viewBlock("taps_shrink").state == Block::State::Created);

    const Tensor coeffs = viewBlock("taps_shrink").outputs.at("coeffs").tensor;
    REQUIRE(coeffs.rank() == 2);
    REQUIRE(coeffs.shape(0) == 1);
    REQUIRE(coeffs.shape(1) == 51);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("taps_shrink", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("heads")) == 1);
    REQUIRE(std::any_cast<std::vector<F32>>(saved.at("center")).size() == 1);
}
