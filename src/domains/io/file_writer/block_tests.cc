#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

namespace {

std::filesystem::path InputPath() {
    auto path = std::filesystem::temp_directory_path() /
                "jst_test_file_writer_input_f32.raw";
    return path;
}

std::filesystem::path OutputPath(const std::string& suffix) {
    auto path = std::filesystem::temp_directory_path() /
                ("jst_test_file_writer_block_" + suffix + ".raw");
    return path;
}

void Cleanup(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture,
                 "FileWriter block without input stays incomplete",
                 "[modules][io][file_writer][block]") {
    Parser::Map config;
    config["filepath"] = OutputPath("no_input").string();
    config["overwrite"] = std::string("true");
    config["recording"] = std::string("true");

    REQUIRE(flowgraph->blockCreate("writer", "file_writer", config, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("writer")->state() ==
            Block::State::Incomplete);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FileWriter block writes after connecting source",
                 "[modules][io][file_writer][block][wiring]") {
    const auto inputPath = InputPath();
    const auto outputPath = OutputPath("connected");
    Cleanup(inputPath);
    Cleanup(outputPath);

    {
        std::ofstream out(inputPath, std::ios::binary);
        F32 values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    Parser::Map readerConfig;
    readerConfig["filepath"] = inputPath.string();
    readerConfig["dataType"] = std::string("F32");
    readerConfig["batchSize"] = std::string("4");
    readerConfig["loop"] = std::string("false");
    REQUIRE(flowgraph->blockCreate("reader", "file_reader", readerConfig, {}) ==
            Result::SUCCESS);

    Parser::Map writerConfig;
    writerConfig["filepath"] = outputPath.string();
    writerConfig["overwrite"] = std::string("true");
    writerConfig["recording"] = std::string("true");

    TensorMap inputs;
    inputs["buffer"].requested("reader", "signal");

    REQUIRE(flowgraph->blockCreate("writer", "file_writer", writerConfig,
                                   inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("writer")->state() == Block::State::Created);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(std::filesystem::exists(outputPath));
    REQUIRE(std::filesystem::file_size(outputPath) == 4 * sizeof(F32));

    REQUIRE(flowgraph->blockDisconnect("writer", "buffer") == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("writer")->state() == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("writer", "buffer", "reader", "signal") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("writer")->state() == Block::State::Created);

    Cleanup(inputPath);
    Cleanup(outputPath);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FileWriter block reconfigure recording toggles in-place",
                 "[modules][io][file_writer][block][reconfigure]") {
    const auto inputPath = InputPath();
    const auto outputPath = OutputPath("toggle");
    Cleanup(inputPath);
    Cleanup(outputPath);

    {
        std::ofstream out(inputPath, std::ios::binary);
        F32 values[2] = {7.0f, 8.0f};
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    Parser::Map readerConfig;
    readerConfig["filepath"] = inputPath.string();
    readerConfig["dataType"] = std::string("F32");
    readerConfig["batchSize"] = std::string("2");
    readerConfig["loop"] = std::string("true");
    REQUIRE(flowgraph->blockCreate("reader", "file_reader", readerConfig, {}) ==
            Result::SUCCESS);

    Parser::Map writerConfig;
    writerConfig["filepath"] = outputPath.string();
    writerConfig["overwrite"] = std::string("true");
    writerConfig["recording"] = std::string("true");

    TensorMap inputs;
    inputs["buffer"].requested("reader", "signal");
    REQUIRE(flowgraph->blockCreate("writer", "file_writer", writerConfig,
                                   inputs) == Result::SUCCESS);

    Parser::Map update;
    update["recording"] = std::string("false");
    REQUIRE(flowgraph->blockReconfigure("writer", update) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("writer")->state() == Block::State::Created);

    Cleanup(inputPath);
    Cleanup(outputPath);
}
