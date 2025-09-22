#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

namespace {

std::filesystem::path TestFilePath(const std::string& suffix) {
    auto path = std::filesystem::temp_directory_path() /
                ("jst_test_file_reader_block_" + suffix + ".raw");
    return path;
}

void Cleanup(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture,
                 "FileReader block creates and exposes output contract",
                 "[modules][io][file_reader][block]") {
    const auto path = TestFilePath("create");
    Cleanup(path);

    {
        std::ofstream file(path, std::ios::binary);
        const std::vector<F32> data = {0.0f, 1.0f, 2.0f, 3.0f};
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(F32)));
    }

    Parser::Map config;
    config["filepath"] = path.string();
    config["dataType"] = std::string("F32");
    config["batchSize"] = std::string("4");

    REQUIRE(flowgraph->blockCreate("reader", "file_reader", config, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("reader")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("reader")
                            ->outputs().at("signal").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.rank() == 1);
    REQUIRE(out.shape(0) == 4);

    Cleanup(path);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FileReader block reconfigure updates without recreate for loop",
                 "[modules][io][file_reader][block][reconfigure]") {
    const auto path = TestFilePath("reconfigure");
    Cleanup(path);

    {
        std::ofstream file(path, std::ios::binary);
        const std::vector<F32> data = {0.0f, 1.0f, 2.0f, 3.0f};
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(F32)));
    }

    Parser::Map config;
    config["filepath"] = path.string();
    config["dataType"] = std::string("F32");
    config["batchSize"] = std::string("4");
    config["loop"] = std::string("true");
    REQUIRE(flowgraph->blockCreate("reader", "file_reader", config, {}) ==
            Result::SUCCESS);

    Parser::Map loopOnly;
    loopOnly["loop"] = std::string("false");
    REQUIRE(flowgraph->blockReconfigure("reader", loopOnly) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("reader")->state() == Block::State::Created);

    Parser::Map resize;
    resize["filepath"] = path.string();
    resize["dataType"] = std::string("F32");
    resize["batchSize"] = std::string("2");
    resize["loop"] = std::string("false");
    REQUIRE(flowgraph->blockReconfigure("reader", resize) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("reader")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("reader")
                            ->outputs().at("signal").tensor;
    REQUIRE(out.shape(0) == 2);

    Cleanup(path);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FileReader block invalid batch size enters error state",
                 "[modules][io][file_reader][block][validation]") {
    Parser::Map config;
    config["batchSize"] = std::string("0");

    REQUIRE(flowgraph->blockCreate("reader_invalid", "file_reader", config,
                                   {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("reader_invalid")->state() ==
            Block::State::Errored);
}
