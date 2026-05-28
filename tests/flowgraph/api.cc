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

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph persistent meta APIs are covered", "[flowgraph][api][meta]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);

    SECTION("raw persistent meta round-trips at flowgraph scope") {
        Parser::Map source;
        source["order"] = U64{7};
        source["label"] = std::string("global");

        REQUIRE(flowgraph->setPersistentMeta("layout", source) == Result::SUCCESS);
        REQUIRE(flowgraph->hasPersistentMeta("layout"));

        Parser::Map restored;
        REQUIRE(flowgraph->getPersistentMeta("layout", restored) == Result::SUCCESS);
        REQUIRE(restored.contains("order"));
        REQUIRE(restored.contains("label"));
        REQUIRE(std::any_cast<U64>(restored.at("order")) == 7);
        REQUIRE(std::any_cast<std::string>(restored.at("label")) == "global");

        Parser::Map tried;
        REQUIRE(flowgraph->tryGetPersistentMeta("layout", tried));
        REQUIRE(std::any_cast<U64>(tried.at("order")) == 7);
    }

    SECTION("typed persistent meta round-trips at block scope") {
        SimpleMetaFixture source;
        source.order = 3;
        source.label = "block";

        REQUIRE(flowgraph->setPersistentMeta("dock", source, "gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->hasPersistentMeta("dock", "gen1"));

        SimpleMetaFixture restored;
        REQUIRE(flowgraph->getPersistentMeta("dock", restored, "gen1") == Result::SUCCESS);
        REQUIRE(restored.order == 3);
        REQUIRE(restored.label == "block");

        SimpleMetaFixture tried;
        REQUIRE(flowgraph->tryGetPersistentMeta("dock", tried, "gen1"));
        REQUIRE(tried.order == 3);
        REQUIRE(tried.label == "block");
    }

    SECTION("missing typed persistent meta leaves the destination unchanged") {
        SimpleMetaFixture restored;
        restored.order = 99;
        restored.label = "keep";

        REQUIRE(flowgraph->getPersistentMeta("missing", restored) == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->hasPersistentMeta("missing"));
        REQUIRE_FALSE(flowgraph->tryGetPersistentMeta("missing", restored));
        REQUIRE(restored.order == 99);
        REQUIRE(restored.label == "keep");
    }

    SECTION("missing raw persistent meta returns success with empty output") {
        Parser::Map restored;
        REQUIRE(flowgraph->getPersistentMeta("missing", restored, "gen1") == Result::SUCCESS);
        REQUIRE(restored.empty());
    }

    SECTION("typed persistent meta must serialize to a map") {
        REQUIRE(flowgraph->setPersistentMeta("invalid", U64{7}) == Result::ERROR);
    }

    SECTION("persistent meta can be cleared") {
        Parser::Map source;
        source["order"] = U64{7};

        REQUIRE(flowgraph->setPersistentMeta("layout", source) == Result::SUCCESS);
        REQUIRE(flowgraph->setPersistentMeta("dock", source, "gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->hasPersistentMeta("layout"));
        REQUIRE(flowgraph->hasPersistentMeta("dock", "gen1"));

        REQUIRE(flowgraph->clearPersistentMeta("layout") == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->hasPersistentMeta("layout"));
        REQUIRE(flowgraph->hasPersistentMeta("dock", "gen1"));

        REQUIRE(flowgraph->clearPersistentMeta("dock", "gen1") == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->hasPersistentMeta("dock", "gen1"));

        REQUIRE(flowgraph->setPersistentMeta("dock", source, "gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->clearAllPersistentMeta() == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->hasPersistentMeta("dock", "gen1"));
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph volatile meta APIs are covered", "[flowgraph][api][volatile]") {
    SECTION("raw volatile meta round-trips") {
        Parser::Map source;
        source["order"] = U64{7};
        source["label"] = std::string("live");

        REQUIRE(flowgraph->setVolatileMeta("layout", source) == Result::SUCCESS);
        REQUIRE(flowgraph->hasVolatileMeta("layout"));

        Parser::Map restored;
        REQUIRE(flowgraph->getVolatileMeta("layout", restored) == Result::SUCCESS);
        REQUIRE(restored.contains("order"));
        REQUIRE(restored.contains("label"));
        REQUIRE(std::any_cast<U64>(restored.at("order")) == 7);
        REQUIRE(std::any_cast<std::string>(restored.at("label")) == "live");

        Parser::Map tried;
        REQUIRE(flowgraph->tryGetVolatileMeta("layout", tried));
        REQUIRE(std::any_cast<U64>(tried.at("order")) == 7);
    }

    SECTION("typed volatile meta round-trips") {
        SimpleMetaFixture source;
        source.order = 3;
        source.label = "live";

        REQUIRE(flowgraph->setVolatileMeta("dock", source) == Result::SUCCESS);
        REQUIRE(flowgraph->hasVolatileMeta("dock"));

        SimpleMetaFixture restored;
        REQUIRE(flowgraph->getVolatileMeta("dock", restored) == Result::SUCCESS);
        REQUIRE(restored.order == 3);
        REQUIRE(restored.label == "live");

        SimpleMetaFixture tried;
        REQUIRE(flowgraph->tryGetVolatileMeta("dock", tried));
        REQUIRE(tried.order == 3);
        REQUIRE(tried.label == "live");
    }

    SECTION("typed volatile meta must serialize to a map") {
        REQUIRE(flowgraph->setVolatileMeta("sampleRate", U64{48000}) == Result::ERROR);
        REQUIRE_FALSE(flowgraph->hasVolatileMeta("sampleRate"));
    }

    SECTION("missing volatile meta leaves the destination unchanged") {
        SimpleMetaFixture restored;
        restored.order = 99;
        restored.label = "keep";

        REQUIRE(flowgraph->getVolatileMeta("missing", restored) == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->hasVolatileMeta("missing"));
        REQUIRE_FALSE(flowgraph->tryGetVolatileMeta("missing", restored));
        REQUIRE(restored.order == 99);
        REQUIRE(restored.label == "keep");
    }

    SECTION("timestamped volatile meta resolves latest matching range") {
        SimpleMetaFixture scan;
        scan.order = 1;
        scan.label = "scan";
        SimpleMetaFixture track;
        track.order = 2;
        track.label = "track";
        SimpleMetaFixture override;
        override.order = 3;
        override.label = "override";

        REQUIRE(flowgraph->setVolatileMeta("mode", scan, 0, 99) == Result::SUCCESS);
        REQUIRE(flowgraph->setVolatileMeta("mode", track, 100, 200) == Result::SUCCESS);
        REQUIRE(flowgraph->setVolatileMeta("mode", override, 50, 175) == Result::SUCCESS);

        SimpleMetaFixture mode;
        REQUIRE(flowgraph->tryGetVolatileMeta("mode", mode, 25));
        REQUIRE(mode.label == "scan");
        REQUIRE(flowgraph->tryGetVolatileMeta("mode", mode, 75));
        REQUIRE(mode.label == "override");
        REQUIRE(flowgraph->tryGetVolatileMeta("mode", mode, 150));
        REQUIRE(mode.label == "override");
        REQUIRE(flowgraph->tryGetVolatileMeta("mode", mode, 190));
        REQUIRE(mode.label == "track");
        REQUIRE_FALSE(flowgraph->hasVolatileMeta("mode", 250));
    }

    SECTION("volatile meta rejects invalid timestamp ranges") {
        Parser::Map source;
        source["value"] = std::string("invalid");

        REQUIRE(flowgraph->setVolatileMeta("mode", source, 10, 9) == Result::ERROR);
        REQUIRE_FALSE(flowgraph->hasVolatileMeta("mode", 10));
    }

    SECTION("volatile meta can be cleared") {
        Parser::Map sampleRate;
        sampleRate["value"] = U64{48000};
        Parser::Map centerFrequency;
        centerFrequency["value"] = F64{915.0e6};

        REQUIRE(flowgraph->setVolatileMeta("sampleRate", sampleRate) == Result::SUCCESS);
        REQUIRE(flowgraph->setVolatileMeta("centerFrequency", centerFrequency) == Result::SUCCESS);
        REQUIRE(flowgraph->hasVolatileMeta("sampleRate"));
        REQUIRE(flowgraph->hasVolatileMeta("centerFrequency"));

        REQUIRE(flowgraph->clearVolatileMeta("sampleRate") == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->hasVolatileMeta("sampleRate"));
        REQUIRE(flowgraph->hasVolatileMeta("centerFrequency"));

        REQUIRE(flowgraph->clearAllVolatileMeta() == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->hasVolatileMeta("centerFrequency"));
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
        REQUIRE(flowgraph->setPersistentMeta("layout", meta) == Result::SUCCESS);
        Parser::Map session;
        session["id"] = U64{42};
        REQUIRE(flowgraph->setVolatileMeta("session", session) == Result::SUCCESS);

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
        REQUIRE(imported.getPersistentMeta("layout", restored) == Result::SUCCESS);
        REQUIRE(restored.order == 12);
        REQUIRE(restored.label == "graph");
        REQUIRE_FALSE(imported.hasVolatileMeta("session"));

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
