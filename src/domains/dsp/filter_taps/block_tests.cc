#include <catch2/catch_test_macros.hpp>

#include <any>
#include <limits>
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

    for (U64 cycle = 0; cycle < 3; ++cycle) {
        REQUIRE(flowgraph->compute() == Result::SUCCESS);
    }

    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("taps", metrics) == Result::SUCCESS);
    REQUIRE(metrics.size() == 1);
    REQUIRE(metrics.front().name == "runtime:filter_taps");
    const auto* timing = std::any_cast<Module::Timing>(&metrics.front().value);
    REQUIRE(timing != nullptr);
    REQUIRE(timing->cycles == 1);
    REQUIRE(timing->computeTime == 0.0f);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter taps retains its interface for zero heads",
                 "[modules][dsp][filter_taps][validation]") {
    Blocks::FilterTaps config;
    config.heads = 0;

    REQUIRE(flowgraph->blockCreate("taps_zero_heads", config, {}) ==
            Result::SUCCESS);

    const auto block = viewBlock("taps_zero_heads");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.interfaceInputs.empty());
    REQUIRE_FALSE(block.interfaceOutputs.empty());
    REQUIRE_FALSE(block.interfaceConfigs.empty());
    REQUIRE(block.outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter taps rejects unrepresentable heads without throwing",
                 "[modules][dsp][filter_taps][validation]") {
    Blocks::FilterTaps config;
    config.heads = std::numeric_limits<U64>::max();

    REQUIRE(flowgraph->blockCreate("taps_huge_heads", config, {}) ==
            Result::SUCCESS);
    const auto block = viewBlock("taps_huge_heads");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
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

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter taps block delegates scalar validation to its module",
                 "[modules][dsp][filter_taps][validation]") {
    Blocks::FilterTaps config;
    config.sampleRate = std::numeric_limits<F32>::quiet_NaN();

    REQUIRE(flowgraph->blockCreate("taps_bad", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("taps_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("taps_bad").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter taps block preserves a module-rejected update for recovery",
                 "[modules][dsp][filter_taps][reconfigure][validation]") {
    Blocks::FilterTaps config;
    config.bandwidth = 0.2e6f;
    config.taps = 5;
    REQUIRE(flowgraph->blockCreate("taps_update", config, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["bandwidth"] = std::numeric_limits<F32>::infinity();
    REQUIRE(flowgraph->blockReconfigure("taps_update", update) == Result::SUCCESS);
    REQUIRE(viewBlock("taps_update").state == Block::State::Errored);
    REQUIRE(viewBlock("taps_update").outputs.empty());

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("taps_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<F32>(saved.at("bandwidth")) ==
            std::numeric_limits<F32>::infinity());
    REQUIRE(std::any_cast<U64>(saved.at("taps")) == config.taps);

    Parser::Map recovery;
    recovery["bandwidth"] = config.bandwidth;
    REQUIRE(flowgraph->blockReconfigure("taps_update", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("taps_update").state == Block::State::Created);
    REQUIRE(viewBlock("taps_update").outputs.contains("coeffs"));
}
