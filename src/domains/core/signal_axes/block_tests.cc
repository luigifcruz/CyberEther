#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>

#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/core/signal_axes/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "jetstream/domains/visualization/lineplot/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal axes block assigns and serializes positional roles",
                 "[modules][signal_axes][block]") {
    Blocks::OnesTensor source;
    source.shape = {2, 3, 4, 5};
    REQUIRE(flowgraph->blockCreate("axes_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("axes_src", "buffer");

    Blocks::SignalAxes config;
    config.axes = "[C, S, B, *]";
    REQUIRE(flowgraph->blockCreate("axes", config, inputs) == Result::SUCCESS);

    const auto block = viewBlock("axes");
    REQUIRE(block.state == Block::State::Created);
    const Tensor& output = block.outputs.at("buffer").tensor;
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == 0);
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 1);
    REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == 2);

    const auto axes = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) {
                                       return entry.name == "axes";
                                   });
    REQUIRE(axes != block.interfaceConfigs.end());
    REQUIRE(axes->format == "text");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("axes", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(saved.at("axes")) == "[C, S, B, *]");
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal axes block reconfigures roles",
                 "[modules][signal_axes][block][reconfigure]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("axes_recfg_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("axes_recfg_src", "window");

    Blocks::SignalAxes config;
    config.axes = "[*]";
    REQUIRE(flowgraph->blockCreate("axes_recfg", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("axes_recfg").outputs.at("buffer").tensor.hasAttribute("sampleAxis"));

    Parser::Map update;
    update["axes"] = std::string("[C]");
    REQUIRE(flowgraph->blockReconfigure("axes_recfg", update) == Result::SUCCESS);

    const Tensor output = viewBlock("axes_recfg").outputs.at("buffer").tensor;
    REQUIRE_FALSE(output.hasAttribute("sampleAxis"));
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == 0);

    update["axes"] = std::string{};
    REQUIRE(flowgraph->blockReconfigure("axes_recfg", update) == Result::SUCCESS);
    const Tensor inheritedOutput =
        viewBlock("axes_recfg").outputs.at("buffer").tensor;
    REQUIRE(std::any_cast<Index>(inheritedOutput.attribute("sampleAxis")) == 0);
    REQUIRE_FALSE(inheritedOutput.hasAttribute("channelAxis"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal axes block enables channel-only visualization",
    "[modules][signal_axes][block][integration]") {
    Blocks::OnesTensor source;
    source.shape = {2, 16};
    REQUIRE(flowgraph->blockCreate("axes_plot_src", source, {}) == Result::SUCCESS);

    TensorMap axesInputs;
    axesInputs["buffer"].requested("axes_plot_src", "buffer");
    Blocks::SignalAxes axesConfig;
    axesConfig.axes = "[B, C]";
    REQUIRE(flowgraph->blockCreate("axes_plot", axesConfig, axesInputs) ==
            Result::SUCCESS);

    const auto axesBlock = viewBlock("axes_plot");
    const Tensor& axesOutput = axesBlock.outputs.at("buffer").tensor;
    REQUIRE_FALSE(axesOutput.hasAttribute("sampleAxis"));
    REQUIRE(std::any_cast<Index>(axesOutput.attribute("batchAxis")) == 0);
    REQUIRE(std::any_cast<Index>(axesOutput.attribute("channelAxis")) == 1);

    TensorMap plotInputs;
    plotInputs["signal"].requested("axes_plot", "buffer");
    REQUIRE(flowgraph->blockCreate("axes_lineplot", Blocks::Lineplot{}, plotInputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("axes_lineplot").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal axes block reports invalid layouts",
                 "[modules][signal_axes][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {2, 8};
    REQUIRE(flowgraph->blockCreate("axes_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("axes_bad_src", "buffer");
    Blocks::SignalAxes config;
    config.axes = "[C, C]";
    REQUIRE(flowgraph->blockCreate("axes_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("axes_bad").state == Block::State::Errored);
}
