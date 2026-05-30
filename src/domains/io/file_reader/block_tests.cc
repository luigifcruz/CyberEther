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

std::string MetricString(const Flowgraph& flowgraph,
                         const std::string& block,
                         const std::string& key) {
    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph.view().metrics(block, metrics) == Result::SUCCESS);

    for (const auto& metric : metrics) {
        if (metric.name == key) {
            return std::any_cast<std::string>(metric.value);
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
    REQUIRE(viewBlock("reader").state == Block::State::Created);

    const Tensor out = viewBlock("reader").outputs.at("signal").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.rank() == 1);
    REQUIRE(out.shape(0) == 4);

    REQUIRE(flowgraph->blockDestroy("reader", false) == Result::SUCCESS);

    Cleanup(path);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FileReader block current bandwidth metric updates after a read",
                 "[modules][io][file_reader][block][metrics]") {
    const auto path = TestFilePath("bandwidth");
    Cleanup(path);

    std::vector<U8> data(8 * 1024 * 1024);
    for (U64 i = 0; i < data.size(); ++i) {
        data[i] = static_cast<U8>(i & 0xFF);
    }
    WriteRawFile(path, data);

    Parser::Map config;
    config["filepath"] = path.string();
    config["dataType"] = std::string("U8");
    config["batchSize"] = std::to_string(data.size());
    config["loop"] = std::string("false");

    REQUIRE(flowgraph->blockCreate("reader", "file_reader", config, {}) ==
            Result::SUCCESS);

    REQUIRE(MetricString(*flowgraph, "reader", "currentBandwidth") == "0.0 MB/s");

    std::this_thread::sleep_for(std::chrono::milliseconds(120));

    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const auto metric = MetricString(*flowgraph, "reader", "currentBandwidth");
    INFO("currentBandwidth=" << metric);
    REQUIRE(metric != "N/A");
    REQUIRE(metric != "0.0 MB/s");

    REQUIRE(flowgraph->blockDestroy("reader", false) == Result::SUCCESS);

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
    REQUIRE(viewBlock("reader").state == Block::State::Created);

    Parser::Map resize;
    resize["filepath"] = path.string();
    resize["dataType"] = std::string("F32");
    resize["batchSize"] = std::string("2");
    resize["loop"] = std::string("false");
    REQUIRE(flowgraph->blockReconfigure("reader", resize) == Result::SUCCESS);
    REQUIRE(viewBlock("reader").state == Block::State::Created);

    const Tensor out = viewBlock("reader").outputs.at("signal").tensor;
    REQUIRE(out.shape(0) == 2);

    REQUIRE(flowgraph->blockDestroy("reader", false) == Result::SUCCESS);

    Cleanup(path);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FileReader block invalid batch size enters error state",
                 "[modules][io][file_reader][block][validation]") {
    Parser::Map config;
    config["batchSize"] = std::string("0");

    REQUIRE(flowgraph->blockCreate("reader_invalid", "file_reader", config,
                                   {}) == Result::SUCCESS);
    REQUIRE(viewBlock("reader_invalid").state ==
            Block::State::Errored);
}
