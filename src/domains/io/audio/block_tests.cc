#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "jetstream/platform.hh"
#include <fstream>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

namespace {

std::filesystem::path AudioInputFilePath() {
    auto path = std::filesystem::temp_directory_path() /
                "jst_test_audio_input_f32.raw";
    return path;
}

void Cleanup(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture,
                 "Audio block connects to an F32 source",
                 "[modules][io][audio][block]") {
    if (Registry::ListAvailableModules("audio").empty()) {
        SUCCEED("Audio module is unavailable in this build.");
        return;
    }

    const auto path = AudioInputFilePath();
    Cleanup(path);

    {
        std::ofstream out(path, std::ios::binary);
        F32 data[128] = {};
        out.write(reinterpret_cast<const char*>(data), sizeof(data));
    }

    Parser::Map readerConfig;
    readerConfig["filepath"] = Platform::PathToUtf8(path);
    readerConfig["dataType"] = std::string("F32");
    readerConfig["batchSize"] = std::string("64");
    readerConfig["loop"] = std::string("true");

    REQUIRE(flowgraph->blockCreate("reader", "file_reader", readerConfig, {}) ==
            Result::SUCCESS);

    TensorMap audioInputs;
    audioInputs["buffer"].requested("reader", "signal");

    REQUIRE(flowgraph->blockCreate("audio_out", "audio", {}, audioInputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("audio_out").state ==
            Block::State::Created);

    REQUIRE(flowgraph->blockDestroy("audio_out", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("reader", false) == Result::SUCCESS);

    Cleanup(path);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Audio block delegates input dtype validation to its module",
                  "[modules][io][audio][block][validation]") {
    if (Registry::ListAvailableModules("audio").empty()) {
        SUCCEED("Audio module is unavailable in this build.");
        return;
    }

    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("64");

    REQUIRE(flowgraph->blockCreate("audio_bad_source", "signal_generator",
                                   sourceConfig, {}) == Result::SUCCESS);

    TensorMap audioInputs;
    audioInputs["buffer"].requested("audio_bad_source", "signal");

    REQUIRE(flowgraph->blockCreate("audio_bad_dtype", "audio", {},
                                   audioInputs) == Result::SUCCESS);
    const auto& block = viewBlock("audio_bad_dtype");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.diagnostic.find("[MODULE_AUDIO_NATIVE_CPU]") !=
            std::string::npos);

    REQUIRE(flowgraph->blockDestroy("audio_bad_dtype", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("audio_bad_source", false) == Result::SUCCESS);
}
