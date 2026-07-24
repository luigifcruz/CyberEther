#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <any>
#include <string>
#include <vector>

#include "jetstream/domains/core/comparator/block.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "jetstream/logger.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Comparator block creates and exposes output",
                 "[modules][comparator][block]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("cmp_src_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("cmp_src_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].requested("cmp_src_a", "window");
    inputs["input1"].requested("cmp_src_b", "window");

    Blocks::Comparator config;
    config.inputCount = 2;
    REQUIRE(flowgraph->blockCreate("cmp_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("cmp_block").state == Block::State::Created);
    REQUIRE(viewBlock("cmp_block").outputs.count("error") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Comparator block rejects input count bounds",
                 "[modules][comparator][block][validation]") {
    Blocks::Comparator config;
    std::string name;

    SECTION("fewer than two inputs") {
        config.inputCount = 1;
        name = "cmp_too_few";
    }
    SECTION("more than sixteen inputs") {
        config.inputCount = 17;
        name = "cmp_too_many";
    }

    REQUIRE(flowgraph->blockCreate(name, config, {}) == Result::SUCCESS);

    const auto block = viewBlock(name);
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.interfaceInputs.empty());
    REQUIRE(block.interfaceOutputs.empty());
    REQUIRE(block.interfaceConfigs.empty());
    REQUIRE(block.outputs.empty());
    REQUIRE(JST_LOG_LAST_ERROR().find("[BLOCK_COMPARATOR]") != std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture, "Comparator block handles disconnect and reconnect",
                 "[modules][comparator][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("cmp_life_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("cmp_life_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].requested("cmp_life_a", "window");
    inputs["input1"].requested("cmp_life_b", "window");

    Blocks::Comparator config;
    config.inputCount = 2;
    REQUIRE(flowgraph->blockCreate("cmp_life", config, inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("cmp_life", "input1") == Result::SUCCESS);
    REQUIRE(viewBlock("cmp_life").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("cmp_life", "input1", "cmp_life_b", "window") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("cmp_life").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture, "Comparator block accepts F32 tolerance reconfigure",
                 "[modules][comparator][block][reconfigure]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("cmp_cfg_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("cmp_cfg_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].requested("cmp_cfg_a", "window");
    inputs["input1"].requested("cmp_cfg_b", "window");

    Blocks::Comparator config;
    REQUIRE(flowgraph->blockCreate("cmp_cfg", config, inputs) == Result::SUCCESS);

    Parser::Map update = viewBlock("cmp_cfg").config;
    update["tolerance"] = F32{0.25f};

    REQUIRE(flowgraph->blockReconfigure("cmp_cfg", update) == Result::SUCCESS);
    REQUIRE(viewBlock("cmp_cfg").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture, "Comparator block delegates tolerance validation",
                 "[modules][comparator][block][validation]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("cmp_bad_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("cmp_bad_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].requested("cmp_bad_a", "window");
    inputs["input1"].requested("cmp_bad_b", "window");

    Blocks::Comparator config;
    config.tolerance = -0.25f;
    REQUIRE(flowgraph->blockCreate("cmp_bad", config, inputs) == Result::SUCCESS);

    const auto block = viewBlock("cmp_bad");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find("[MODULE_COMPARATOR]") != std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Comparator block preserves tolerance and compute after rejected update",
                 "[modules][comparator][block][reconfigure][validation]") {
    Blocks::SignalGenerator sourceA;
    sourceA.signalType = "dc";
    sourceA.amplitude = 0.0f;
    sourceA.bufferSize = 4;
    REQUIRE(flowgraph->blockCreate("cmp_update_a", sourceA, {}) == Result::SUCCESS);

    Blocks::SignalGenerator sourceB = sourceA;
    sourceB.amplitude = 0.25f;
    REQUIRE(flowgraph->blockCreate("cmp_update_b", sourceB, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].requested("cmp_update_a", "signal");
    inputs["input1"].requested("cmp_update_b", "signal");

    Blocks::Comparator config;
    config.tolerance = 0.5f;
    REQUIRE(flowgraph->blockCreate("cmp_update", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map rejected;
    rejected["tolerance"] = F32{-0.25f};
    REQUIRE(flowgraph->blockReconfigure("cmp_update", rejected) == Result::ERROR);

    const auto block = viewBlock("cmp_update");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.interfaceInputs.size() == config.inputCount);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("cmp_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<F32>(saved.at("tolerance")) == config.tolerance);

    Tensor output = block.outputs.at("error").tensor;
    std::fill(output.data<F32>(), output.data<F32>() + output.size(), -1.0f);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    for (U64 i = 0; i < output.size(); ++i) {
        REQUIRE_THAT(output.at<F32>(i),
                     Catch::Matchers::WithinAbs(0.25f, 1e-6f));
    }

    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("cmp_update", metrics) == Result::SUCCESS);
    const auto match = std::find_if(metrics.begin(), metrics.end(), [](const auto& metric) {
        return metric.name == "match";
    });
    REQUIRE(match != metrics.end());
    REQUIRE(std::any_cast<std::string>(match->value) == "PASS");
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Comparator block shrinks disconnected dynamic topology",
                 "[modules][comparator][block][reconfigure][topology]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("cmp_shrink_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("cmp_shrink_b", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("cmp_shrink_c", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].requested("cmp_shrink_a", "window");
    inputs["input1"].requested("cmp_shrink_b", "window");
    inputs["input2"].requested("cmp_shrink_c", "window");

    Blocks::Comparator config;
    config.inputCount = 3;
    REQUIRE(flowgraph->blockCreate("cmp_shrink", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("cmp_shrink").interfaceInputs.size() == 3);

    Parser::Map update;
    update["inputCount"] = U64{2};
    REQUIRE(flowgraph->blockReconfigure("cmp_shrink", update) == Result::ERROR);
    REQUIRE(viewBlock("cmp_shrink").state == Block::State::Created);
    REQUIRE(viewBlock("cmp_shrink").inputs.size() == 3);

    REQUIRE(flowgraph->blockDisconnect("cmp_shrink", "input2") == Result::SUCCESS);
    REQUIRE(viewBlock("cmp_shrink").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockReconfigure("cmp_shrink", update) == Result::SUCCESS);

    const auto block = viewBlock("cmp_shrink");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.interfaceInputs.size() == 2);
    REQUIRE(block.inputs.size() == 2);
    REQUIRE_FALSE(block.inputs.contains("input2"));
    REQUIRE(block.outputs.contains("error"));
}
