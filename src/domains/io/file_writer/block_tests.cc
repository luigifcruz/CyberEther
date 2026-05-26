#include <catch2/catch_test_macros.hpp>

#include <any>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "jetstream/block_interface.hh"

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

std::string MetricString(const std::shared_ptr<Block>& block, const std::string& key) {
    for (const auto& [metricKey, entry] : block->interface()->metrics()) {
        if (metricKey == key) {
            return std::any_cast<std::string>(entry.metric());
        }
    }

    FAIL("Missing metric: " << key);
    return {};
}

template<typename T>
void WriteRawFile(const std::filesystem::path& path, const std::vector<T>& data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(T)));
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

    REQUIRE(flowgraph->blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("reader", false) == Result::SUCCESS);

    Cleanup(inputPath);
    Cleanup(outputPath);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FileWriter block current bandwidth metric updates after a write",
                 "[modules][io][file_writer][block][metrics]") {
    const auto inputPath = OutputPath("bandwidth_input");
    const auto outputPath = OutputPath("bandwidth_output");
    Cleanup(inputPath);
    Cleanup(outputPath);

    std::vector<U8> data(8 * 1024 * 1024);
    for (U64 i = 0; i < data.size(); ++i) {
        data[i] = static_cast<U8>(i & 0xFF);
    }
    WriteRawFile(inputPath, data);

    Parser::Map readerConfig;
    readerConfig["filepath"] = inputPath.string();
    readerConfig["dataType"] = std::string("U8");
    readerConfig["batchSize"] = std::to_string(data.size());
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

    const auto block = flowgraph->blockList().at("writer");
    REQUIRE(MetricString(block, "currentBandwidth") == "0.0 MB/s");

    std::this_thread::sleep_for(std::chrono::milliseconds(120));

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(std::filesystem::file_size(outputPath) == data.size());

    const auto metric = MetricString(block, "currentBandwidth");
    INFO("currentBandwidth=" << metric);
    REQUIRE(metric != "N/A");
    REQUIRE(metric != "0.0 MB/s");

    REQUIRE(flowgraph->blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("reader", false) == Result::SUCCESS);

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

    REQUIRE(flowgraph->blockDestroy("writer", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("reader", false) == Result::SUCCESS);

    Cleanup(inputPath);
    Cleanup(outputPath);
}
