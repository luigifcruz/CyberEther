#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include <jetstream/flowgraph_environment.hh>
#include <jetstream/flowgraph_metadata.hh>

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

bool ContainsKey(const std::vector<std::string>& keys, const std::string& key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

void CreateSerializableGraph(Flowgraph& flowgraph) {
    REQUIRE(flowgraph.blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph.blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap addInputs;
    addInputs["a"].requested("gen1", "signal");
    addInputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph.blockCreate("add1", "add", {}, addInputs) == Result::SUCCESS);
    REQUIRE(ViewBlock(flowgraph, "add1").state == Block::State::Created);
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
        REQUIRE(flowgraph->view().has("gen1"));
        REQUIRE(viewBlock("gen1").type == "signal_generator");

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

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph metadata APIs are covered", "[flowgraph][api][metadata]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);

    SECTION("raw metadata round-trips at flowgraph scope") {
        Parser::Map source;
        source["order"] = U64{7};
        source["label"] = std::string("global");

        REQUIRE(flowgraph->metadata().set("layout", source) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().has("layout"));

        Parser::Map restored;
        REQUIRE(flowgraph->metadata().get("layout", restored) == Result::SUCCESS);
        REQUIRE(restored.contains("order"));
        REQUIRE(restored.contains("label"));
        REQUIRE(std::any_cast<U64>(restored.at("order")) == 7);
        REQUIRE(std::any_cast<std::string>(restored.at("label")) == "global");

        Parser::Map tried;
        REQUIRE(flowgraph->metadata().tryGet("layout", tried));
        REQUIRE(std::any_cast<U64>(tried.at("order")) == 7);
    }

    SECTION("typed metadata round-trips at block scope") {
        SimpleMetaFixture source;
        source.order = 3;
        source.label = "block";

        REQUIRE(flowgraph->metadata().set("dock", source, "gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().has("dock", "gen1"));

        SimpleMetaFixture restored;
        REQUIRE(flowgraph->metadata().get("dock", restored, "gen1") == Result::SUCCESS);
        REQUIRE(restored.order == 3);
        REQUIRE(restored.label == "block");

        SimpleMetaFixture tried;
        REQUIRE(flowgraph->metadata().tryGet("dock", tried, "gen1"));
        REQUIRE(tried.order == 3);
        REQUIRE(tried.label == "block");
    }

    SECTION("missing typed metadata leaves the destination unchanged") {
        SimpleMetaFixture restored;
        restored.order = 99;
        restored.label = "keep";

        REQUIRE(flowgraph->metadata().get("missing", restored) == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->metadata().has("missing"));
        REQUIRE_FALSE(flowgraph->metadata().tryGet("missing", restored));
        REQUIRE(restored.order == 99);
        REQUIRE(restored.label == "keep");
    }

    SECTION("missing raw metadata returns success with empty output") {
        Parser::Map restored;
        REQUIRE(flowgraph->metadata().get("missing", restored, "gen1") == Result::SUCCESS);
        REQUIRE(restored.empty());
    }

    SECTION("metadata keys can be listed") {
        Parser::Map global;
        global["order"] = U64{7};
        Parser::Map block;
        block["order"] = U64{3};

        REQUIRE(flowgraph->metadata().set("layout", global) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("dock", block, "gen1") == Result::SUCCESS);

        std::vector<std::string> globalKeys;
        REQUIRE(flowgraph->metadata().keys(globalKeys) == Result::SUCCESS);
        REQUIRE(ContainsKey(globalKeys, "layout"));
        REQUIRE_FALSE(ContainsKey(globalKeys, "dock"));

        std::vector<std::string> blockKeys;
        REQUIRE(flowgraph->metadata().keys(blockKeys, "gen1") == Result::SUCCESS);
        REQUIRE(ContainsKey(blockKeys, "dock"));
        REQUIRE_FALSE(ContainsKey(blockKeys, "layout"));
    }

    SECTION("typed metadata must serialize to a map") {
        REQUIRE(flowgraph->metadata().set("invalid", U64{7}) == Result::ERROR);
    }

    SECTION("metadata can be cleared") {
        Parser::Map source;
        source["order"] = U64{7};

        REQUIRE(flowgraph->metadata().set("layout", source) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("dock", source, "gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().has("layout"));
        REQUIRE(flowgraph->metadata().has("dock", "gen1"));

        REQUIRE(flowgraph->metadata().clear("layout") == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->metadata().has("layout"));
        REQUIRE(flowgraph->metadata().has("dock", "gen1"));

        REQUIRE(flowgraph->metadata().clear("dock", "gen1") == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->metadata().has("dock", "gen1"));

        REQUIRE(flowgraph->metadata().set("dock", source, "gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().clearAll() == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->metadata().has("dock", "gen1"));
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph environment APIs are covered", "[flowgraph][api][environment]") {
    SECTION("raw environment value round-trips") {
        Parser::Map source;
        source["order"] = U64{7};
        source["label"] = std::string("live");

        REQUIRE(flowgraph->environment().set("layout", source) == Result::SUCCESS);
        REQUIRE(flowgraph->environment().has("layout"));

        Parser::Map restored;
        REQUIRE(flowgraph->environment().get("layout", restored) == Result::SUCCESS);
        REQUIRE(restored.contains("order"));
        REQUIRE(restored.contains("label"));
        REQUIRE(std::any_cast<U64>(restored.at("order")) == 7);
        REQUIRE(std::any_cast<std::string>(restored.at("label")) == "live");

        Parser::Map tried;
        REQUIRE(flowgraph->environment().tryGet("layout", tried));
        REQUIRE(std::any_cast<U64>(tried.at("order")) == 7);
    }

    SECTION("typed environment value round-trips") {
        SimpleMetaFixture source;
        source.order = 3;
        source.label = "live";

        REQUIRE(flowgraph->environment().set("dock", source) == Result::SUCCESS);
        REQUIRE(flowgraph->environment().has("dock"));

        SimpleMetaFixture restored;
        REQUIRE(flowgraph->environment().get("dock", restored) == Result::SUCCESS);
        REQUIRE(restored.order == 3);
        REQUIRE(restored.label == "live");

        SimpleMetaFixture tried;
        REQUIRE(flowgraph->environment().tryGet("dock", tried));
        REQUIRE(tried.order == 3);
        REQUIRE(tried.label == "live");
    }

    SECTION("typed environment value must serialize to a map") {
        REQUIRE(flowgraph->environment().set("sampleRate", U64{48000}) == Result::ERROR);
        REQUIRE_FALSE(flowgraph->environment().has("sampleRate"));
    }

    SECTION("missing environment value leaves the destination unchanged") {
        SimpleMetaFixture restored;
        restored.order = 99;
        restored.label = "keep";

        REQUIRE(flowgraph->environment().get("missing", restored) == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->environment().has("missing"));
        REQUIRE_FALSE(flowgraph->environment().tryGet("missing", restored));
        REQUIRE(restored.order == 99);
        REQUIRE(restored.label == "keep");
    }

    SECTION("timestamped environment value resolves latest matching range") {
        SimpleMetaFixture scan;
        scan.order = 1;
        scan.label = "scan";
        SimpleMetaFixture track;
        track.order = 2;
        track.label = "track";
        SimpleMetaFixture override;
        override.order = 3;
        override.label = "override";

        REQUIRE(flowgraph->environment().set("mode", scan, 0, 99) == Result::SUCCESS);
        REQUIRE(flowgraph->environment().set("mode", track, 100, 200) == Result::SUCCESS);
        REQUIRE(flowgraph->environment().set("mode", override, 50, 175) == Result::SUCCESS);

        SimpleMetaFixture mode;
        REQUIRE(flowgraph->environment().tryGet("mode", mode, 25));
        REQUIRE(mode.label == "scan");
        REQUIRE(flowgraph->environment().tryGet("mode", mode, 75));
        REQUIRE(mode.label == "override");
        REQUIRE(flowgraph->environment().tryGet("mode", mode, 150));
        REQUIRE(mode.label == "override");
        REQUIRE(flowgraph->environment().tryGet("mode", mode, 190));
        REQUIRE(mode.label == "track");
        REQUIRE_FALSE(flowgraph->environment().has("mode", 250));
    }

    SECTION("environment rejects invalid timestamp ranges") {
        Parser::Map source;
        source["value"] = std::string("invalid");

        REQUIRE(flowgraph->environment().set("mode", source, 10, 9) == Result::ERROR);
        REQUIRE_FALSE(flowgraph->environment().has("mode", 10));
    }

    SECTION("environment values can be cleared") {
        Parser::Map sampleRate;
        sampleRate["value"] = U64{48000};
        Parser::Map centerFrequency;
        centerFrequency["value"] = F64{915.0e6};

        REQUIRE(flowgraph->environment().set("sampleRate", sampleRate) == Result::SUCCESS);
        REQUIRE(flowgraph->environment().set("centerFrequency", centerFrequency) == Result::SUCCESS);
        REQUIRE(flowgraph->environment().has("sampleRate"));
        REQUIRE(flowgraph->environment().has("centerFrequency"));

        REQUIRE(flowgraph->environment().clear("sampleRate") == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->environment().has("sampleRate"));
        REQUIRE(flowgraph->environment().has("centerFrequency"));

        REQUIRE(flowgraph->environment().clearAll() == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->environment().has("centerFrequency"));
    }

    SECTION("environment keys can be listed") {
        Parser::Map sampleRate;
        sampleRate["value"] = U64{48000};
        Parser::Map mode;
        mode["value"] = std::string("scan");

        REQUIRE(flowgraph->environment().set("sampleRate", sampleRate) == Result::SUCCESS);
        REQUIRE(flowgraph->environment().set("mode", mode, 100, 200) == Result::SUCCESS);

        std::vector<std::string> keys;
        REQUIRE(flowgraph->environment().keys(keys) == Result::SUCCESS);
        REQUIRE(ContainsKey(keys, "sampleRate"));
        REQUIRE(ContainsKey(keys, "mode"));
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
        REQUIRE(flowgraph->metadata().set("layout", meta) == Result::SUCCESS);
        Parser::Map session;
        session["id"] = U64{42};
        REQUIRE(flowgraph->environment().set("session", session) == Result::SUCCESS);

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
        REQUIRE(imported.view().size() == 3);

        SimpleMetaFixture restored;
        REQUIRE(imported.metadata().get("layout", restored) == Result::SUCCESS);
        REQUIRE(restored.order == 12);
        REQUIRE(restored.label == "graph");
        REQUIRE_FALSE(imported.environment().has("session"));

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
        REQUIRE(flowgraph->view().size() == 3);
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
        REQUIRE(viewBlock("gen2").state == Block::State::Created);
        REQUIRE(viewBlock("add1").state == Block::State::Created);

        std::vector<char> exported;
        REQUIRE(flowgraph->exportToBlob(exported) == Result::SUCCESS);
        const std::string yaml(exported.begin(), exported.end());
        REQUIRE(yaml.starts_with("---\nversion: 2\n"));
        REQUIRE(yaml.find("protocolVersion") == std::string::npos);
    }
}
