#include <catch2/catch_test_macros.hpp>

#include <any>

#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/core/slice/block.hh"
#include "jetstream/domains/core/throttle/block.hh"
#include "jetstream/domains/dsp/spectrum_engine/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "jetstream/logger.hh"
#include "jetstream/registry.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Slice block creates with contiguous output",
                 "[modules][slice][block]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("slice_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_src", "window");

    Blocks::Slice config;
    config.slice = "[0:8]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("slice_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("slice_block").state == Block::State::Created);
    REQUIRE(viewBlock("slice_block").outputs.count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Slice block keeps contiguous copies on CUDA",
                 "[modules][slice][block][CUDA]") {
    if (Registry::ListAvailableModules("window", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("slice", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("duplicate", DeviceType::CUDA).empty()) {
        SUCCEED("Required CUDA modules are unavailable in this build.");
        return;
    }

    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("slice_cuda_src", source, {}, DeviceType::CUDA) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_cuda_src", "window");

    Blocks::Slice config;
    config.slice = "[0:8]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("slice_cuda", config, inputs, DeviceType::CUDA) ==
            Result::SUCCESS);

    const auto block = viewBlock("slice_cuda");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.device() == DeviceType::CUDA);
}

TEST_CASE_METHOD(FlowgraphFixture, "Slice block propagates module slice validation",
                  "[modules][slice][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("slice_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_bad_src", "window");

    Blocks::Slice config;
    config.slice = "[::0]";
    REQUIRE(flowgraph->blockCreate("slice_bad", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("slice_bad");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find("[MODULE_SLICE]") != std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture, "Slice block preserves candidate state after invalid update",
                 "[modules][slice][block][reconfigure][validation]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("slice_update_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_update_src", "window");

    Blocks::Slice config;
    config.slice = "[0:16:2]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("slice_update", config, inputs) == Result::SUCCESS);

    REQUIRE(viewBlock("slice_update").outputs.at("buffer").tensor.shape() == Shape{8});

    Parser::Map rejected;
    rejected["slice"] = std::string("[..., ...]");
    REQUIRE(flowgraph->blockReconfigure("slice_update", rejected) == Result::SUCCESS);

    const auto errored = viewBlock("slice_update");
    REQUIRE(errored.state == Block::State::Errored);
    REQUIRE(errored.outputs.empty());
    REQUIRE(errored.interfaceInputs.size() == 1);
    REQUIRE(errored.interfaceOutputs.size() == 1);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("slice_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(saved.at("slice")) == "[..., ...]");
    REQUIRE(std::any_cast<bool>(saved.at("contiguous")) == config.contiguous);

    Parser::Map recovery;
    recovery["slice"] = config.slice;
    REQUIRE(flowgraph->blockReconfigure("slice_update", recovery) == Result::SUCCESS);
    const auto recovered = viewBlock("slice_update");
    REQUIRE(recovered.state == Block::State::Created);
    REQUIRE(recovered.outputs.at("buffer").tensor.shape() == Shape{8});
    REQUIRE(recovered.outputs.at("buffer").tensor.contiguous());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Slice block changes a rank-three index without diagnostics",
                 "[modules][slice][block][reconfigure]") {
    Blocks::OnesTensor source;
    source.shape = {8, 2, 2024};
    REQUIRE(flowgraph->blockCreate("slice_rank3_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_rank3_src", "buffer");

    Blocks::Slice config;
    config.slice = "[:, 0, :]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("slice_rank3", config, inputs) ==
            Result::SUCCESS);

    JST_LOG_LAST_ERROR().clear();
    Parser::Map update;
    update["slice"] = std::string("[:, 1, :]");
    REQUIRE(flowgraph->blockReconfigure("slice_rank3", update) ==
            Result::SUCCESS);
    REQUIRE(JST_LOG_LAST_ERROR().empty());

    const auto output = viewBlock("slice_rank3").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{8, 2024});
    REQUIRE(output.contiguous());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Slice reconfiguration preserves invalid downstream interfaces for recovery",
                 "[modules][slice][block][reconfigure][lifecycle]") {
    Blocks::OnesTensor source;
    source.shape = {8, 2, 16};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("slice_lifecycle_src", source, {}) ==
            Result::SUCCESS);

    Blocks::Slice sliceConfig;
    sliceConfig.slice = "[:,0,:]";
    TensorMap sliceInputs;
    sliceInputs["buffer"].requested("slice_lifecycle_src", "buffer");
    REQUIRE(flowgraph->blockCreate("slice_lifecycle", sliceConfig, sliceInputs) ==
            Result::SUCCESS);

    Blocks::SpectrumEngine spectrumConfig;
    spectrumConfig.axis = 1;
    spectrumConfig.enableScale = true;
    spectrumConfig.rangeMin = -100.0f;
    spectrumConfig.rangeMax = -10.0f;
    TensorMap spectrumInputs;
    spectrumInputs["buffer"].requested("slice_lifecycle", "buffer");
    REQUIRE(flowgraph->blockCreate("slice_lifecycle_spectrum",
                                   spectrumConfig,
                                   spectrumInputs) == Result::SUCCESS);

    TensorMap throttleInputs;
    throttleInputs["buffer"].requested("slice_lifecycle_spectrum", "buffer");
    REQUIRE(flowgraph->blockCreate("slice_lifecycle_throttle",
                                   Blocks::Throttle{},
                                   throttleInputs) == Result::SUCCESS);
    REQUIRE(viewBlock("slice_lifecycle").state == Block::State::Created);
    REQUIRE(viewBlock("slice_lifecycle_spectrum").state == Block::State::Created);
    REQUIRE(viewBlock("slice_lifecycle_throttle").state == Block::State::Created);

    Parser::Map sliceUpdate;
    sliceUpdate["slice"] = std::string("[0,:,0]");
    REQUIRE(flowgraph->blockReconfigure("slice_lifecycle", sliceUpdate) ==
            Result::SUCCESS);

    const auto slice = viewBlock("slice_lifecycle");
    REQUIRE(slice.state == Block::State::Created);
    REQUIRE(slice.outputs.at("buffer").tensor.shape() == Shape{2});

    const auto spectrum = viewBlock("slice_lifecycle_spectrum");
    REQUIRE(spectrum.state == Block::State::Errored);
    Blocks::SpectrumEngine candidate;
    REQUIRE(candidate.deserialize(spectrum.config) == Result::SUCCESS);
    REQUIRE(candidate.axis == 1);
    REQUIRE_FALSE(candidate.enableAgc);
    REQUIRE(candidate.enableScale);
    REQUIRE(candidate.rangeMin == -100.0f);
    REQUIRE(candidate.rangeMax == -10.0f);
    REQUIRE(spectrum.diagnostic.find("[BLOCK_SPECTRUM_ENGINE]") !=
            std::string::npos);
    REQUIRE(spectrum.interfaceInputs.size() == 1);
    REQUIRE(spectrum.interfaceInputs.at(0).name == "buffer");
    REQUIRE(spectrum.interfaceOutputs.size() == 1);
    REQUIRE(spectrum.interfaceOutputs.at(0).name == "buffer");
    REQUIRE(spectrum.interfaceConfigs.size() == 5);
    REQUIRE(spectrum.interfaceConfigs.at(0).name == "axis");
    REQUIRE(spectrum.interfaceConfigs.at(1).name == "enableAgc");
    REQUIRE(spectrum.interfaceConfigs.at(2).name == "enableScale");
    REQUIRE(spectrum.interfaceConfigs.at(3).name == "rangeMin");
    REQUIRE(spectrum.interfaceConfigs.at(4).name == "rangeMax");
    REQUIRE(spectrum.inputs.at("buffer").resolved());
    REQUIRE(spectrum.inputs.at("buffer").external.has_value());
    REQUIRE(spectrum.inputs.at("buffer").external->block == "slice_lifecycle");
    REQUIRE(spectrum.inputs.at("buffer").external->port == "buffer");

    const auto throttle = viewBlock("slice_lifecycle_throttle");
    REQUIRE(throttle.state == Block::State::Incomplete);
    REQUIRE_FALSE(throttle.inputs.at("buffer").resolved());
    REQUIRE(throttle.inputs.at("buffer").external.has_value());
    REQUIRE(throttle.inputs.at("buffer").external->block ==
            "slice_lifecycle_spectrum");
    REQUIRE(throttle.inputs.at("buffer").external->port == "buffer");

    Parser::Map spectrumUpdate;
    spectrumUpdate["axis"] = I64{0};
    REQUIRE(flowgraph->blockReconfigure("slice_lifecycle_spectrum", spectrumUpdate) ==
            Result::SUCCESS);

    const auto recoveredSpectrum = viewBlock("slice_lifecycle_spectrum");
    REQUIRE(recoveredSpectrum.state == Block::State::Created);
    REQUIRE(recoveredSpectrum.inputs.at("buffer").resolved());
    REQUIRE(recoveredSpectrum.inputs.at("buffer").external.has_value());
    REQUIRE(recoveredSpectrum.inputs.at("buffer").external->block ==
            "slice_lifecycle");
    REQUIRE(recoveredSpectrum.inputs.at("buffer").external->port == "buffer");

    const auto recoveredThrottle = viewBlock("slice_lifecycle_throttle");
    REQUIRE(recoveredThrottle.state == Block::State::Created);
    REQUIRE(recoveredThrottle.inputs.at("buffer").resolved());
    REQUIRE(recoveredThrottle.inputs.at("buffer").external.has_value());
    REQUIRE(recoveredThrottle.inputs.at("buffer").external->block ==
            "slice_lifecycle_spectrum");
    REQUIRE(recoveredThrottle.inputs.at("buffer").external->port == "buffer");
}
