#include <catch2/catch_test_macros.hpp>

#include <filesystem>
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
    readerConfig["filepath"] = path.string();
    readerConfig["dataType"] = std::string("F32");
    readerConfig["batchSize"] = std::string("64");
    readerConfig["loop"] = std::string("true");

    REQUIRE(flowgraph->blockCreate("reader", "file_reader", readerConfig, {}) ==
            Result::SUCCESS);

    TensorMap audioInputs;
    audioInputs["buffer"] = {"reader", "signal", {}};

    REQUIRE(flowgraph->blockCreate("audio_out", "audio", {}, audioInputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("audio_out")->state() ==
            Block::State::Created);

    Cleanup(path);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Audio block rejects invalid sample rate configuration",
                 "[modules][io][audio][block][validation]") {
    if (Registry::ListAvailableModules("audio").empty()) {
        SUCCEED("Audio module is unavailable in this build.");
        return;
    }

    const auto path = AudioInputFilePath();
    Cleanup(path);

    {
        std::ofstream out(path, std::ios::binary);
        F32 data[64] = {};
        out.write(reinterpret_cast<const char*>(data), sizeof(data));
    }

    Parser::Map readerConfig;
    readerConfig["filepath"] = path.string();
    readerConfig["dataType"] = std::string("F32");
    readerConfig["batchSize"] = std::string("64");
    readerConfig["loop"] = std::string("true");

    REQUIRE(flowgraph->blockCreate("reader", "file_reader", readerConfig, {}) ==
            Result::SUCCESS);

    TensorMap audioInputs;
    audioInputs["buffer"] = {"reader", "signal", {}};

    Parser::Map config;
    config["inSampleRate"] = std::string("0");

    const auto createResult = flowgraph->blockCreate("audio_invalid", "audio",
                                                      config, audioInputs);
    REQUIRE(createResult == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("audio_invalid")->state() ==
            Block::State::Errored);

    Cleanup(path);
}
