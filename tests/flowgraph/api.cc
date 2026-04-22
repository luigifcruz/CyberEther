#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>

#include "common.hh"

using namespace Jetstream;
using namespace TestFlowgraph;

namespace {

struct TempFlowgraphFile {
    explicit TempFlowgraphFile(const std::string& name) {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("cyberether-flowgraph-" + name + "-" + std::to_string(nonce) + ".yaml");
    }

    ~TempFlowgraphFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    std::filesystem::path path;
};

void CreateSerializableGraph(Flowgraph& flowgraph) {
    REQUIRE(flowgraph.blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph.blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap addInputs;
    addInputs["a"].requested("gen1", "signal");
    addInputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph.blockCreate("add1", "add", {}, addInputs) == Result::SUCCESS);
    REQUIRE(flowgraph.blockList().at("add1")->state() == Block::State::Created);
}

}  // namespace

TEST_CASE("Flowgraph lifecycle APIs behave consistently", "[flowgraph][api][lifecycle]") {
    Flowgraph flowgraph;

    SECTION("default state exposes empty metadata and no-op runtime calls") {
        REQUIRE(flowgraph.title().empty());
        REQUIRE(flowgraph.summary().empty());
        REQUIRE(flowgraph.author().empty());
        REQUIRE(flowgraph.license().empty());
        REQUIRE(flowgraph.description().empty());
        REQUIRE(flowgraph.path().empty());

        REQUIRE(flowgraph.start() == Result::SUCCESS);
        REQUIRE(flowgraph.stop() == Result::SUCCESS);
        REQUIRE(flowgraph.compute() == Result::SUCCESS);
        REQUIRE(flowgraph.present() == Result::SUCCESS);
        REQUIRE(flowgraph.destroy() == Result::ERROR);
    }

    SECTION("create and destroy update lifecycle state") {
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        REQUIRE(flowgraph.metrics().empty());
        REQUIRE(flowgraph.compute() == Result::SUCCESS);
        REQUIRE(flowgraph.present() == Result::SUCCESS);
        REQUIRE(flowgraph.start() == Result::SUCCESS);
        REQUIRE(flowgraph.stop() == Result::SUCCESS);
        REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::ERROR);
        REQUIRE(flowgraph.destroy() == Result::SUCCESS);
        REQUIRE(flowgraph.path().empty());
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph setter and getter APIs are covered", "[flowgraph][api]") {
    REQUIRE(flowgraph->path().empty());
    REQUIRE(flowgraph->metrics().empty());

    REQUIRE(flowgraph->setTitle("Example") == Result::SUCCESS);
    REQUIRE(flowgraph->setSummary("Summary") == Result::SUCCESS);
    REQUIRE(flowgraph->setAuthor("Author") == Result::SUCCESS);
    REQUIRE(flowgraph->setLicense("MIT") == Result::SUCCESS);
    REQUIRE(flowgraph->setDescription("Description") == Result::SUCCESS);

    REQUIRE(flowgraph->title() == "Example");
    REQUIRE(flowgraph->summary() == "Summary");
    REQUIRE(flowgraph->author() == "Author");
    REQUIRE(flowgraph->license() == "MIT");
    REQUIRE(flowgraph->description() == "Description");
}

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph block configuration APIs are covered", "[flowgraph][api][config]") {
    SECTION("blockCreate accepts Block::Config overload") {
        Blocks::SignalGenerator config;
        config.bufferSize = 4096;
        config.frequency = 2000.0f;

        REQUIRE(flowgraph->blockCreate("gen1", config, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().contains("gen1"));
        REQUIRE(flowgraph->blockList().at("gen1")->config().type() == "signal_generator");

        Parser::Map encoded;
        REQUIRE(flowgraph->blockConfig("gen1", encoded) == Result::SUCCESS);
        REQUIRE(encoded.contains("bufferSize"));
        REQUIRE(encoded.contains("frequency"));
        REQUIRE(std::any_cast<U64>(encoded.at("bufferSize")) == 4096);
        REQUIRE(std::any_cast<F32>(encoded.at("frequency")) == Catch::Approx(2000.0f));
    }

    SECTION("blockConfig fails for missing blocks") {
        Parser::Map encoded;
        REQUIRE(flowgraph->blockConfig("missing", encoded) == Result::ERROR);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph metadata APIs are covered", "[flowgraph][api][meta]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);

    SECTION("raw metadata round-trips at flowgraph scope") {
        Parser::Map source;
        source["order"] = U64{7};
        source["label"] = std::string("global");

        REQUIRE(flowgraph->setMeta("layout", source) == Result::SUCCESS);

        Parser::Map restored;
        REQUIRE(flowgraph->getMeta("layout", restored) == Result::SUCCESS);
        REQUIRE(restored.contains("order"));
        REQUIRE(restored.contains("label"));
        REQUIRE(std::any_cast<U64>(restored.at("order")) == 7);
        REQUIRE(std::any_cast<std::string>(restored.at("label")) == "global");
    }

    SECTION("typed metadata round-trips at block scope") {
        SimpleMetaFixture source;
        source.order = 3;
        source.label = "block";

        REQUIRE(flowgraph->setMeta("dock", source, "gen1") == Result::SUCCESS);

        SimpleMetaFixture restored;
        REQUIRE(flowgraph->getMeta("dock", restored, "gen1") == Result::SUCCESS);
        REQUIRE(restored.order == 3);
        REQUIRE(restored.label == "block");
    }

    SECTION("missing typed metadata leaves the destination unchanged") {
        SimpleMetaFixture restored;
        restored.order = 99;
        restored.label = "keep";

        REQUIRE(flowgraph->getMeta("missing", restored) == Result::SUCCESS);
        REQUIRE(restored.order == 99);
        REQUIRE(restored.label == "keep");
    }

    SECTION("missing raw metadata returns success with empty output") {
        Parser::Map restored;
        REQUIRE(flowgraph->getMeta("missing", restored, "gen1") == Result::SUCCESS);
        REQUIRE(restored.empty());
    }

    SECTION("typed metadata must serialize to a map") {
        REQUIRE(flowgraph->setMeta("invalid", U64{7}) == Result::ERROR);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph file APIs are covered", "[flowgraph][api][file]") {
    SECTION("exportToFile rejects empty paths") {
        REQUIRE(flowgraph->exportToFile("") == Result::ERROR);
    }

    SECTION("importFromFile rejects missing files") {
        const TempFlowgraphFile tempFile("missing");
        REQUIRE(flowgraph->importFromFile(tempFile.path.string()) == Result::ERROR);
        REQUIRE(flowgraph->path() == tempFile.path.string());
    }

    SECTION("exportToFile and importFromFile round-trip a flowgraph") {
        CreateSerializableGraph(*flowgraph);
        REQUIRE(flowgraph->setTitle("File Graph") == Result::SUCCESS);
        REQUIRE(flowgraph->setSummary("File Summary") == Result::SUCCESS);
        REQUIRE(flowgraph->setAuthor("File Author") == Result::SUCCESS);
        REQUIRE(flowgraph->setLicense("BSD-3-Clause") == Result::SUCCESS);
        REQUIRE(flowgraph->setDescription("File Description") == Result::SUCCESS);

        SimpleMetaFixture meta;
        meta.order = 12;
        meta.label = "graph";
        REQUIRE(flowgraph->setMeta("layout", meta) == Result::SUCCESS);

        const TempFlowgraphFile tempFile("roundtrip");
        REQUIRE(flowgraph->exportToFile(tempFile.path.string()) == Result::SUCCESS);
        REQUIRE(flowgraph->path() == tempFile.path.string());
        REQUIRE(std::filesystem::exists(tempFile.path));

        std::ifstream file(tempFile.path, std::ios::binary);
        REQUIRE(file.good());
        const std::string yaml((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        REQUIRE(yaml.starts_with("---\nversion: 2\n"));
        REQUIRE(yaml.find("summary: File Summary") != std::string::npos);
        REQUIRE(yaml.find("license: BSD-3-Clause") != std::string::npos);
        REQUIRE(yaml.find("description: File Description") != std::string::npos);

        Flowgraph imported;
        REQUIRE(imported.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        REQUIRE(imported.importFromFile(tempFile.path.string()) == Result::SUCCESS);
        REQUIRE(imported.path() == tempFile.path.string());
        REQUIRE(imported.title() == "File Graph");
        REQUIRE(imported.summary() == "File Summary");
        REQUIRE(imported.author() == "File Author");
        REQUIRE(imported.license() == "BSD-3-Clause");
        REQUIRE(imported.description() == "File Description");
        REQUIRE(imported.blockList().size() == 3);

        SimpleMetaFixture restored;
        REQUIRE(imported.getMeta("layout", restored) == Result::SUCCESS);
        REQUIRE(restored.order == 12);
        REQUIRE(restored.label == "graph");

        REQUIRE(imported.destroy() == Result::SUCCESS);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph blob APIs are covered", "[flowgraph][api][serialization]") {
    SECTION("importFromBlob rejects unsupported versions") {
        const std::string yaml = R"(---
version: 3
graph: []
)";

        const std::vector<char> blob(yaml.begin(), yaml.end());
        REQUIRE(flowgraph->importFromBlob(blob) == Result::ERROR);
    }

    SECTION("importFromBlob migrates version 1.0.0 graphs") {
        const std::string legacyYaml = R"(---
protocolVersion: 1.0.0
title: Legacy Flowgraph
summary: Legacy Summary
author: Legacy Author
license: Legacy License
description: Legacy Description
graph:
  gen1:
    module: signal_generator
  gen2:
    module: signal_generator
  add1:
    module: add
    input:
      a: '${graph.gen1.output.signal}'
      b: '${graph.gen2.output.signal}'
)";

        const std::vector<char> blob(legacyYaml.begin(), legacyYaml.end());
        REQUIRE(flowgraph->importFromBlob(blob) == Result::SUCCESS);
        REQUIRE(flowgraph->title() == "Legacy Flowgraph");
        REQUIRE(flowgraph->summary() == "Legacy Summary");
        REQUIRE(flowgraph->author() == "Legacy Author");
        REQUIRE(flowgraph->license() == "Legacy License");
        REQUIRE(flowgraph->description() == "Legacy Description");
        REQUIRE(flowgraph->blockList().size() == 3);
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("gen2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);

        std::vector<char> exported;
        REQUIRE(flowgraph->exportToBlob(exported) == Result::SUCCESS);
        const std::string yaml(exported.begin(), exported.end());
        REQUIRE(yaml.starts_with("---\nversion: 2\n"));
        REQUIRE(yaml.find("protocolVersion") == std::string::npos);
    }
}
