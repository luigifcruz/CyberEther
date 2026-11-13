#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>

#include <nlohmann/json.hpp>

#include "jetstream/agent/tools.hh"
#include "jetstream/domains/core/slice/block.hh"
#include "jetstream/registry.hh"
#include "jetstream/flowgraph_view.hh"

using namespace Jetstream;

namespace {

struct GraphHost : Agent::Tools::Host {
    std::shared_ptr<Flowgraph> graph;
    std::vector<std::string> folders;
    std::vector<std::shared_ptr<Flowgraph>> createdGraphs;

    explicit GraphHost(std::shared_ptr<Flowgraph> graph) : graph(std::move(graph)) {}

    bool has(const std::string&) override {
        return false;
    }

    Result create(const std::string&, std::shared_ptr<Flowgraph>& created) override {
        if (folders.empty()) {
            return Result::ERROR;
        }

        created = std::make_shared<Flowgraph>();
        if (created->create({}, nullptr, nullptr, nullptr) != Result::SUCCESS) {
            return Result::ERROR;
        }
        createdGraphs.push_back(created);
        return Result::SUCCESS;
    }

    std::vector<std::string> flowgraphFolders() override {
        return folders;
    }
};

Agent::Tools::Output ExecuteTool(const std::shared_ptr<Flowgraph>& flowgraph, const std::string& name,
                                 const std::string& args) {
    GraphHost host(flowgraph);
    return Agent::Tools(host).execute(host.graph, name, args);
}

std::string FlowgraphNameToCreate(const std::shared_ptr<Flowgraph>& flowgraph, const std::string& args) {
    GraphHost host(flowgraph);
    return Agent::Tools(host).prepare(host.graph, "apply_flowgraph_patch", args).create;
}

}  // namespace

TEST_CASE("Agent exposes the consolidated tool surface", "[agent][tools]") {
    const auto& tools = Agent::Tools::Definitions();
    const std::set<std::string> expected = {
        "search_blocks",
        "get_block_docs",
        "inspect_flowgraph",
        "inspect_block",
        "apply_flowgraph_patch",
        "patch_block_text",
        "read_block_text",
        "find_connection_options",
        "validate_flowgraph",
        "inspect_environment",
        "inspect_block_view",
        "list_examples",
        "get_example",
        "open_example",
        "list_flowgraphs",
        "open_flowgraph",
    };

    std::set<std::string> actual;
    for (const auto& definition : tools) {
        const auto& tool = definition.function;
        actual.insert(tool.name);

        const auto schema = nlohmann::json::parse(tool.parametersJson);
        REQUIRE(schema["type"] == "object");
        REQUIRE(schema["additionalProperties"] == false);

        if (tool.name == "apply_flowgraph_patch") {
            REQUIRE(schema["properties"].contains("flowgraph_name"));
            REQUIRE_FALSE(schema["properties"].contains("create_flowgraph"));
            REQUIRE(schema["properties"]["operations"].contains("description"));
            for (const auto& variant : schema["properties"]["operations"]["items"]["oneOf"]) {
                REQUIRE(variant["properties"]["op"]["const"] != "patch_block_text");
            }
        }
        if (tool.name == "patch_block_text") {
            REQUIRE(schema["required"] == nlohmann::json::array({"name", "field", "patch"}));
            REQUIRE(definition.flags.needsGraph);
            REQUIRE(definition.flags.reportsLayout);
            REQUIRE(definition.flags.mutates);
            REQUIRE(definition.flags.destructive);
            REQUIRE_FALSE(definition.flags.needsCapture);
        }
        if (tool.name == "inspect_block_view") {
            REQUIRE(tool.description.find("image_width") != std::string::npos);
            REQUIRE(tool.description.find("image_height") != std::string::npos);
        }
    }

    REQUIRE(actual == expected);
}

TEST_CASE("Removed agent tools cannot be dispatched", "[agent][tools]") {
    const auto result = ExecuteTool(nullptr, "create_block", R"({"type":"signal_generator"})");
    const auto body = nlohmann::json::parse(result.content);

    REQUIRE(body["error"] == "Unknown tool 'create_block'.");

    const auto phantom = nlohmann::json::parse(ExecuteTool(nullptr, "%s", R"({"%s":"%s"})").content);
    const auto error = phantom["error"].get<std::string>();
    REQUIRE(error.starts_with("Unknown tool '%s'."));
    REQUIRE(error.find("cut apart") != std::string::npos);

    const auto namespaced = nlohmann::json::parse(ExecuteTool(nullptr, "functions.read_block_text", "{}").content);
    REQUIRE(namespaced["error"] == "Unknown tool 'functions.read_block_text'.");
}

TEST_CASE("Agent tools reject malformed argument roots", "[agent][tools]") {
    const auto malformed = ExecuteTool(nullptr, "search_blocks", "{");
    REQUIRE(nlohmann::json::parse(malformed.content).contains("error"));

    const auto array = ExecuteTool(nullptr, "search_blocks", "[]");
    REQUIRE(nlohmann::json::parse(array.content).contains("error"));
}

TEST_CASE("Graph tools report a missing graph", "[agent][tools]") {
    const auto result = ExecuteTool(nullptr, "inspect_flowgraph", "{}");
    const auto body = nlohmann::json::parse(result.content);

    REQUIRE(body["error"] == "No flowgraph is selected.");
}

TEST_CASE("Flowgraph patches only create a graph when none is focused", "[agent][tools][flowgraph]") {
    const std::string patch = R"({
        "flowgraph_name": "new_graph",
        "operations": [{"op": "delete_block", "name": "old"}]
    })";

    REQUIRE(FlowgraphNameToCreate(nullptr, patch) == "new_graph");

    const auto focused = std::make_shared<Flowgraph>();
    REQUIRE(FlowgraphNameToCreate(focused, patch).empty());
}

TEST_CASE("Every optional patch field is validated before any graph mutation", "[agent][tools][validation]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const auto fields = {"name", "ref", "device", "runtime", "provider"};
    const auto invalids = {nlohmann::json(123), nlohmann::json(nullptr), nlohmann::json::array()};

    for (const char* field : fields) {
        for (const auto& invalid : invalids) {
            CAPTURE(field, invalid);

            const nlohmann::json valid = {
                {"op", "create_block"},
                {"type", "signal_generator"},
                {"name", "must_not_exist"},
            };
            const nlohmann::json operation = {
                {"op", "create_block"},
                {"type", "signal_generator"},
                {field, invalid},
            };
            const auto args = nlohmann::json({
                {"operations", nlohmann::json::array({valid, operation})},
            }).dump();

            const auto result = nlohmann::json::parse(ExecuteTool(graph, "apply_flowgraph_patch", args).content);

            CHECK(result.contains("error"));
            CHECK_FALSE(graph->view().has("must_not_exist"));
            CHECK(FlowgraphNameToCreate(nullptr, args).empty());
        }
    }

    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Composite blocks are discoverable and creatable", "[agent][tools][flowgraph]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    const auto search = nlohmann::json::parse(
        ExecuteTool(nullptr, "search_blocks", R"({"query":"filter"})").content);
    const auto filter = std::find_if(search["matches"].begin(), search["matches"].end(), [](const auto& match) {
        return match["type"] == "filter";
    });
    REQUIRE(filter != search["matches"].end());
    REQUIRE((*filter)["implementations"][0]["kind"] == "composite");

    const auto docs = nlohmann::json::parse(
        ExecuteTool(nullptr, "get_block_docs", R"({"type":"filter"})").content);
    REQUIRE(docs["implementations"][0]["kind"] == "composite");
    REQUIRE(docs.contains("default_config"));

    auto flowgraph = std::make_shared<Flowgraph>();
    REQUIRE(flowgraph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const auto patched = nlohmann::json::parse(ExecuteTool(
        flowgraph, "apply_flowgraph_patch", R"({
            "operations": [
                {"op":"create_block","type":"signal_generator","name":"source","ref":"source",
                 "config":{"signalDataType":"CF32","bufferSize":8192,"sampleRate":2000000.0}},
                {"op":"create_block","type":"filter","name":"channel_filter",
                 "config":{"sampleRate":2000000.0,"bandwidth":1000000.0,"center":[0.0],"taps":101}},
                {"op":"connect","source":"$source","output":"signal","block":"channel_filter","input":"signal"},
                {"op":"create_block","type":"decimator","name":"audio_decimator",
                 "config":{"axis":0,"ratio":4}},
                {"op":"connect","source":"$source","output":"signal","block":"audio_decimator","input":"buffer"}
            ]
        })").content);
    REQUIRE(patched["status"] == "success");
    REQUIRE(patched["health"]["valid"] == true);
    REQUIRE(flowgraph->view().has("channel_filter"));
    REQUIRE(flowgraph->view().has("audio_decimator"));

    REQUIRE(flowgraph->blockDestroy("channel_filter", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("audio_decimator", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph->destroy() == Result::SUCCESS);
}

TEST_CASE("Flowgraph patches create blocks from structured JSON", "[agent][tools][flowgraph]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    const auto docsResult = ExecuteTool(nullptr, "get_block_docs", R"({"type":"signal_generator"})");
    const auto docs = nlohmann::json::parse(docsResult.content);
    REQUIRE(docs["default_config"].contains("bufferSize"));
    REQUIRE(docs["observed_interfaces"].empty());

    auto flowgraph = std::make_shared<Flowgraph>();
    REQUIRE(flowgraph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const auto result = ExecuteTool(flowgraph, "apply_flowgraph_patch", R"({
        "operations": [
            {
                "op": "create_block",
                "ref": "source",
                "type": "signal_generator",
                "name": "source",
                "config": {"bufferSize": 1024, "frequency": 1200.5}
            }
        ]
    })");
    const auto body = nlohmann::json::parse(result.content);

    REQUIRE(body["status"] == "success");
    REQUIRE(body["results"][0]["name"] == "source");
    CHECK(body["results"][0]["config"]["bufferSize"] == 1024);
    CHECK(body["results"][0]["config"]["frequency"] == 1200.5);
    CHECK(body["results"][0]["config"]["sampleRate"] == docs["default_config"]["sampleRate"]);
    REQUIRE(body["health"].contains("issues"));
    REQUIRE(flowgraph->view().has("source"));

    const auto summary = nlohmann::json::parse(
        ExecuteTool(flowgraph, "inspect_flowgraph", R"({"detail":"summary"})").content);
    REQUIRE(summary["blocks"][0].contains("config"));
    REQUIRE_FALSE(summary["blocks"][0].contains("description"));

    Flowgraph::View::BlockData block;
    REQUIRE(flowgraph->view().block("source", block) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(block.config.at("bufferSize")) == 1024);
    REQUIRE(std::any_cast<F32>(block.config.at("frequency")) == 1200.5f);

    TensorMap selectorInputs;
    selectorInputs["buffer"].requested("source", "signal");
    Blocks::Slice firstSelector;
    firstSelector.slice = "[0:1]";
    Blocks::Slice secondSelector;
    secondSelector.slice = "[1:2]";
    REQUIRE(flowgraph->blockCreate("first_selector", firstSelector, selectorInputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("second_selector", secondSelector, selectorInputs) == Result::SUCCESS);

    TensorMap sinkInputs;
    sinkInputs["buffer"].requested("first_selector", "buffer");
    Blocks::Slice sinkConfig;
    REQUIRE(flowgraph->blockCreate("sink", sinkConfig, sinkInputs) == Result::SUCCESS);

    const auto routingResult = ExecuteTool(
        flowgraph, "inspect_flowgraph", R"({"detail":"routing","sink":"sink"})");
    const auto routing = nlohmann::json::parse(routingResult.content);

    REQUIRE(routing["sink"] == "sink");
    REQUIRE(routing["alternative_branches"].size() == 1);
    REQUIRE(routing["alternative_branches"][0]["replaces"] == "first_selector");
    REQUIRE(routing["alternative_branches"][0]["candidate"] == "second_selector");
    REQUIRE(routing["alternative_branches"][0]["current_config"]["slice"] == "[0:1]");
    REQUIRE(routing["alternative_branches"][0]["candidate_config"]["slice"] == "[1:2]");
    REQUIRE(routing["alternative_branches"][0]["switch_patch"]["block"] == "sink");

    const auto reconfigured = nlohmann::json::parse(ExecuteTool(
        flowgraph, "apply_flowgraph_patch", R"({"operations":[
            {"op":"reconfigure_block","name":"first_selector","config":{"contiguous":true}},
            {"op":"reconfigure_block","name":"source","config":{"frequency":2400.5}}
        ]})").content);

    REQUIRE(reconfigured["status"] == "success");
    REQUIRE(reconfigured["results"][0]["config"]["contiguous"] == true);
    REQUIRE(reconfigured["results"][1]["config"]["frequency"] == 2400.5);
    REQUIRE_FALSE(reconfigured["results"][1]["config"].contains("bufferSize"));

    const auto source = nlohmann::json::parse(
        ExecuteTool(flowgraph, "inspect_block", R"({"name":"source"})").content);
    REQUIRE(source["config"]["bufferSize"] == 1024);

    const auto partialResult = ExecuteTool(flowgraph, "apply_flowgraph_patch", R"({
        "operations": [
            {"op": "create_block", "ref": "other", "type": "signal_generator", "name": "other"},
            {"op": "connect", "source": "$other", "output": "signal", "block": "missing", "input": "signal"}
        ]
    })");
    const auto partial = nlohmann::json::parse(partialResult.content);

    REQUIRE(partialResult.error);
    REQUIRE(partial["status"] == "partial");
    REQUIRE(partial["block_count"] == 5);
    REQUIRE(partial["health"].contains("issues"));

    const std::string invalidPatch = R"({
        "operations": [{"op": "create_block", "type": "signal_generator", "config": []}]
    })";
    REQUIRE(FlowgraphNameToCreate(nullptr, invalidPatch).empty());

    const auto invalidConfig = nlohmann::json::parse(
        ExecuteTool(flowgraph, "apply_flowgraph_patch", invalidPatch).content);
    REQUIRE(invalidConfig["error"].get<std::string>().find("create_block") != std::string::npos);

    REQUIRE(flowgraph->blockDestroy("other", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("sink", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("first_selector", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("second_selector", false) == Result::SUCCESS);
    REQUIRE(flowgraph->blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph->destroy() == Result::SUCCESS);
}

TEST_CASE("Agent tools read and write node position and size", "[agent][tools][flowgraph]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto flowgraph = std::make_shared<Flowgraph>();
    REQUIRE(flowgraph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const auto created = nlohmann::json::parse(ExecuteTool(flowgraph, "apply_flowgraph_patch", R"({
        "operations": [{"op": "create_block", "type": "signal_generator", "name": "source"}]
    })").content);
    REQUIRE(created["status"] == "success");
    REQUIRE_FALSE(created["results"][0]["node"].contains("width"));
    REQUIRE_FALSE(created["results"][0]["node"].contains("height"));
    REQUIRE(created["results"][0]["layout_status"] == "pending");
    REQUIRE(created["results"][0]["bounds"].is_null());

    // Node metadata is reported even when no layout has been persisted yet.
    const auto summary = nlohmann::json::parse(
        ExecuteTool(flowgraph, "inspect_flowgraph", R"({"detail":"summary"})").content);
    REQUIRE(summary["blocks"][0]["node"]["x"].get<F32>() == 0.0f);
    REQUIRE_FALSE(summary["blocks"][0]["node"].contains("width"));
    REQUIRE(summary["blocks"][0]["layout_status"] == "pending");
    REQUIRE(summary["blocks"][0]["bounds"].is_null());

    // Positioning does not force an explicit size or invent measured bounds.
    const auto positioned = nlohmann::json::parse(ExecuteTool(flowgraph, "apply_flowgraph_patch",
        R"({"operations":[{"op":"set_node_layout","name":"source","x":20,"y":30}]})").content);
    REQUIRE(positioned["results"][0]["node"]["x"] == 20.0f);
    REQUIRE_FALSE(positioned["results"][0]["node"].contains("width"));
    REQUIRE_FALSE(positioned["results"][0]["node"].contains("height"));
    REQUIRE(positioned["results"][0]["bounds"].is_null());

    // Move and resize in a single patch, matching the persisted flowgraph file.
    const auto patched = nlohmann::json::parse(ExecuteTool(flowgraph, "apply_flowgraph_patch", R"({
        "operations": [
            {"op": "set_node_layout", "name": "source", "x": 120.0, "y": 240.0, "width": 320.0, "height": 480.0}
        ]
    })").content);
    REQUIRE(patched["status"] == "success");
    REQUIRE(patched["results"][0]["node"]["x"].get<F32>() == 120.0f);
    REQUIRE(patched["results"][0]["node"]["y"].get<F32>() == 240.0f);
    REQUIRE(patched["results"][0]["node"]["width"].get<F32>() == 320.0f);
    REQUIRE(patched["results"][0]["node"]["height"].get<F32>() == 480.0f);

    // Partial updates leave the other fields untouched.
    const auto moved = nlohmann::json::parse(ExecuteTool(flowgraph, "apply_flowgraph_patch", R"({
        "operations": [{"op": "set_node_layout", "name": "source", "x": 50.0}]
    })").content);
    REQUIRE(moved["status"] == "success");
    REQUIRE(moved["results"][0]["node"]["x"].get<F32>() == 50.0f);
    REQUIRE(moved["results"][0]["node"]["y"].get<F32>() == 240.0f);
    REQUIRE(moved["results"][0]["node"]["width"].get<F32>() == 320.0f);

    const auto block = nlohmann::json::parse(
        ExecuteTool(flowgraph, "inspect_block", R"({"name":"source"})").content);
    INFO(block.dump());
    REQUIRE_FALSE(block.contains("error"));
    REQUIRE(block["node"]["x"].get<F32>() == 50.0f);
    REQUIRE(block["node"]["height"].get<F32>() == 480.0f);
    REQUIRE(block["layout_status"] == "pending");
    REQUIRE(block["bounds"].is_null());

    // Layout edits require at least one field and reject non-numeric values.
    const auto missing = nlohmann::json::parse(ExecuteTool(flowgraph, "apply_flowgraph_patch",
        R"({"operations":[{"op":"set_node_layout","name":"source"}]})").content);
    REQUIRE(missing["error"].get<std::string>().find("set_node_layout") != std::string::npos);

    const auto badType = nlohmann::json::parse(ExecuteTool(flowgraph, "apply_flowgraph_patch",
        R"({"operations":[{"op":"set_node_layout","name":"source","x":"left"}]})").content);
    REQUIRE(badType["error"].get<std::string>().find("must be a number") != std::string::npos);

    REQUIRE(flowgraph->blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph->destroy() == Result::SUCCESS);
}

TEST_CASE("Text patches require unique matches after anchors", "[agent][tools][text]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    std::string text = "def f():\n    value = 1\n    value = 1\n";
    std::string patch;
    std::string expected = text;
    bool ambiguous = true;

    SECTION("Unanchored duplicate statements are ambiguous") {
        patch = "@@\n-    value = 1\n+    value = 2";
    }
    SECTION("Anchored duplicate statements are ambiguous") {
        patch = "@@ def f():\n-    value = 1\n+    value = 2";
    }
    SECTION("Anchored duplicate statements remain ambiguous after whitespace normalization") {
        patch = "@@ def f():\n-value = 1\n+value = 2";
    }
    SECTION("An anchor can exclude earlier matches") {
        text = "value = 1\ndef f():\n    value = 1\n";
        patch = "@@ def f():\n-value = 1\n+value = 2";
        expected = "value = 1\ndef f():\n    value = 2\n";
        ambiguous = false;
    }

    const nlohmann::json create = {{"operations", nlohmann::json::array({
        {{"op", "create_block"}, {"type", "note"}, {"name", "note"}, {"config", {{"content", text}}}},
    })}};
    REQUIRE_FALSE(ExecuteTool(graph, "apply_flowgraph_patch", create.dump()).error);

    const nlohmann::json edit = {{"operations", nlohmann::json::array({
        {{"op", "patch_block_text"}, {"name", "note"}, {"field", "content"}, {"patch", patch}},
    })}};
    const auto result = ExecuteTool(graph, "apply_flowgraph_patch", edit.dump());
    CHECK(result.error == ambiguous);

    if (ambiguous) {
        const auto body = nlohmann::json::parse(result.content);
        REQUIRE(body["results"][0].contains("error"));

        const auto error = body["results"][0]["error"].get<std::string>();
        CHECK(error.find("matches 2 places (lines 2, 3)") != std::string::npos);
    }

    const auto note = nlohmann::json::parse(
        ExecuteTool(graph, "inspect_block", R"({"name":"note"})").content);
    CHECK(note["config"]["content"] == expected);

    REQUIRE(graph->blockDestroy("note", false) == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Text patch insertions preserve intended newlines", "[agent][tools][text]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const struct {
        std::string text;
        std::string patch;
        std::string expected;
        U64 lineCount;
    } cases[] = {
        {"", "@@\n+hello", "hello", 1},
        {"", "@@\n+hello\n+world", "hello\nworld", 2},
        {"", "@@\n+hello\n*** End of File", "hello", 1},
        {"", "@@\n+hello\n@@\n+world", "hello\nworld", 2},
        {"\n", "@@\n+hello", "\nhello\n", 2},
        {"existing", "@@\n+hello", "existing\nhello", 2},
        {"existing\n", "@@\n+hello", "existing\nhello\n", 2},
    };

    for (const auto& test : cases) {
        CAPTURE(test.text, test.patch);

        const nlohmann::json create = {{"operations", nlohmann::json::array({
            {{"op", "create_block"}, {"type", "note"}, {"name", "note"}, {"config", {{"content", test.text}}}},
        })}};
        REQUIRE_FALSE(ExecuteTool(graph, "apply_flowgraph_patch", create.dump()).error);

        const nlohmann::json edit = {{"operations", nlohmann::json::array({
            {{"op", "patch_block_text"}, {"name", "note"}, {"field", "content"}, {"patch", test.patch}},
        })}};
        const auto result = ExecuteTool(graph, "apply_flowgraph_patch", edit.dump());
        REQUIRE_FALSE(result.error);

        const auto body = nlohmann::json::parse(result.content);
        CHECK(body["results"][0]["line_count"] == test.lineCount);
        if (test.text.empty()) {
            CHECK(body["results"][0]["changes"][0]["line"] == 1);
        }

        const auto note = nlohmann::json::parse(
            ExecuteTool(graph, "inspect_block", R"({"name":"note"})").content);
        CHECK(note["config"]["content"] == test.expected);

        REQUIRE(graph->blockDestroy("note", false) == Result::SUCCESS);
    }

    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Full block inspection preserves numeric arrays and UTF-8 text",
          "[agent][tools][flowgraph]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    nlohmann::json centers = nlohmann::json::array();
    for (int i = 0; i < 20; ++i) {
        centers.push_back(i * 1000.0);
    }
    const std::string text = jst::fmt::format("{}é日本語", std::string(255, 'x'));

    const nlohmann::json patch = {{"operations", nlohmann::json::array({
        {{"op", "create_block"}, {"type", "filter_taps"}, {"name", "taps"},
         {"config", {{"heads", 20}, {"center", centers}}}},
        {{"op", "create_block"}, {"type", "note"}, {"name", "note"}, {"config", {{"content", text}}}},
    })}};
    const auto created = ExecuteTool(graph, "apply_flowgraph_patch", patch.dump());
    REQUIRE_FALSE(created.error);

    const auto inspected = nlohmann::json::parse(
        ExecuteTool(graph, "inspect_block", R"({"name":"taps"})").content);
    CHECK(inspected["config"]["center"] == centers);

    const auto note = nlohmann::json::parse(
        ExecuteTool(graph, "inspect_block", R"({"name":"note"})").content);
    CHECK(note["config"]["content"] == text);

    const auto summary = ExecuteTool(graph, "inspect_flowgraph", "{}");
    CHECK_FALSE(summary.error);
    CHECK_NOTHROW(nlohmann::json::parse(summary.content));

    REQUIRE(graph->blockDestroy("note", false) == Result::SUCCESS);
    REQUIRE(graph->blockDestroy("taps", false) == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Patch refs accept the $ref: spelling and are checked before any mutation",
          "[agent][tools][validation]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const auto apply = [&](const nlohmann::json& operations) {
        const nlohmann::json args = {{"operations", operations}};
        return nlohmann::json::parse(ExecuteTool(graph, "apply_flowgraph_patch", args.dump()).content);
    };

    SECTION("the alias names the block created with that ref") {
        const auto body = apply(nlohmann::json::array({
            {{"op", "create_block"}, {"type", "note"}, {"name", "source"}, {"ref", "src"}},
            {{"op", "set_node_layout"}, {"name", "$ref:src"}, {"x", 10}, {"y", 20}},
        }));
        REQUIRE(body["status"] == "success");
        CHECK(body["results"][1]["name"] == "source");
    }

    SECTION("an exact label wins over the alias reading") {
        const auto body = apply(nlohmann::json::array({
            {{"op", "create_block"}, {"type", "note"}, {"name", "aliased"}, {"ref", "ref:a"}},
            {{"op", "set_node_layout"}, {"name", "$ref:a"}, {"x", 1}, {"y", 1}},
        }));
        REQUIRE(body["status"] == "success");
        CHECK(body["results"][1]["name"] == "aliased");
    }

    SECTION("an undefined ref rejects the whole patch before anything runs") {
        for (const std::string bad : {"$ref:0", "$soapy"}) {
            CAPTURE(bad);
            const nlohmann::json operations = nlohmann::json::array({
                {{"op", "create_block"}, {"type", "note"}, {"name", "must_not_exist"}},
                {{"op", "create_block"}, {"type", "note"}, {"name", "other"}, {"ref", "src"}},
                {{"op", "connect"}, {"source", bad}, {"output", "signal"}, {"block", "$src"}, {"input", "signal"}},
            });
            const auto body = apply(operations);

            REQUIRE(body.contains("error"));
            const auto error = body["error"].get<std::string>();
            CHECK(error.find("uses ref '" + bad + "'") != std::string::npos);
            CHECK(error.find("$src") != std::string::npos);
            CHECK(error.ends_with("Nothing was changed."));
            CHECK_FALSE(graph->view().has("must_not_exist"));
            CHECK_FALSE(graph->view().has("other"));
            CHECK(FlowgraphNameToCreate(nullptr, nlohmann::json({{"operations", operations}}).dump()).empty());
        }
    }

    SECTION("a ref used before its create_block is rejected") {
        const auto body = apply(nlohmann::json::array({
            {{"op", "set_node_layout"}, {"name", "$late"}, {"x", 1}, {"y", 1}},
            {{"op", "create_block"}, {"type", "note"}, {"name", "late_block"}, {"ref", "late"}},
        }));
        REQUIRE(body.contains("error"));
        CHECK(body["error"].get<std::string>().find("No earlier create_block sets a ref") != std::string::npos);
        CHECK_FALSE(graph->view().has("late_block"));
    }

    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Flowgraph patches accept operations sent as JSON text", "[agent][tools][validation]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const auto run = [&](const nlohmann::json& operations) {
        const nlohmann::json args = {{"operations", operations}};
        return ExecuteTool(graph, "apply_flowgraph_patch", args.dump());
    };

    SECTION("a complete array sent as text is applied") {
        const std::string text = R"([{"op":"create_block","type":"note","name":"from_text"}])";
        const auto result = run(text);
        REQUIRE_FALSE(result.error);
        CHECK(graph->view().has("from_text"));

        const nlohmann::json create = {{"flowgraph_name", "new_graph"}, {"operations", text}};
        CHECK(FlowgraphNameToCreate(nullptr, create.dump()) == "new_graph");
    }

    SECTION("a missing closing bracket is repaired") {
        const auto result = run(R"([{"op":"create_block","type":"note","name":"repaired"})");
        REQUIRE_FALSE(result.error);
        CHECK(graph->view().has("repaired"));
    }

    SECTION("a call cut off inside a string explains the likely cause") {
        const std::string text = R"([{"op":"create_block","type":"note","name":"cut","config":{"content":"closes with )";
        const auto result = run(text);
        REQUIRE(result.error);

        const auto error = nlohmann::json::parse(result.content)["error"].get<std::string>();
        CHECK(error.find("not a valid JSON array") != std::string::npos);
        CHECK(error.find("</parameter>") != std::string::npos);
        CHECK_FALSE(graph->view().has("cut"));
        CHECK(FlowgraphNameToCreate(nullptr, nlohmann::json({{"operations", text}}).dump()).empty());
    }

    SECTION("an empty array and other types keep the plain message") {
        for (const nlohmann::json operations : {nlohmann::json("[]"), nlohmann::json(5)}) {
            const auto body = nlohmann::json::parse(run(operations).content);
            CHECK(body["error"] == "'operations' must be a non-empty array.");
        }
    }

    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("The flat text patch tool edits one text field", "[agent][tools][text]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const std::string text = "def f():\n    value = 1\n    other = 3\n    value = 1\n";
    const nlohmann::json create = {{"operations", nlohmann::json::array({
        {{"op", "create_block"}, {"type", "note"}, {"name", "note"}, {"config", {{"content", text}}}},
    })}};
    REQUIRE_FALSE(ExecuteTool(graph, "apply_flowgraph_patch", create.dump()).error);

    const auto content = [&] {
        return nlohmann::json::parse(ExecuteTool(graph, "inspect_block", R"({"name":"note"})").content)
            ["config"]["content"].get<std::string>();
    };
    const auto patch = [&](const nlohmann::json& args) {
        return ExecuteTool(graph, "patch_block_text", args.dump());
    };

    SECTION("a strict patch applies and reports the change") {
        const auto result = patch({{"name", "note"}, {"field", "content"},
                                   {"patch", "@@\n def f():\n-    value = 1\n+    value = 2\n     other = 3"}});
        REQUIRE_FALSE(result.error);

        const auto body = nlohmann::json::parse(result.content);
        CHECK(body["status"] == "success");
        CHECK(body["name"] == "note");
        CHECK(body["field"] == "content");
        CHECK(body["line_count"] == 4);
        CHECK(body["changes"][0]["removed"] == 1);
        CHECK(body["changes"][0]["added"] == 1);
        CHECK(body.contains("state"));
        CHECK(body["health"].contains("issues"));
        CHECK(body.contains("layout_status"));
        CHECK_FALSE(body.contains("note"));
        CHECK(content() == "def f():\n    value = 2\n    other = 3\n    value = 1\n");
    }

    SECTION("failures change nothing and say so") {
        const auto ambiguous = patch({{"name", "note"}, {"field", "content"},
                                      {"patch", "@@\n-    value = 1\n+    value = 2"}});
        REQUIRE(ambiguous.error);
        const auto error = nlohmann::json::parse(ambiguous.content)["error"].get<std::string>();
        CHECK(error.find("matches 2 places") != std::string::npos);
        CHECK(error.ends_with("Nothing was changed."));
        CHECK(content() == text);

        const auto mistyped = nlohmann::json::parse(
            patch({{"name", "note"}, {"field", "content"}, {"patch", 5}}).content);
        CHECK(mistyped["error"] == "'patch' must be a string.");

        const auto missing = nlohmann::json::parse(
            patch({{"name", "nope"}, {"field", "content"}, {"patch", "@@\n+x"}}).content);
        CHECK(missing["error"].get<std::string>().find("not found") != std::string::npos);

        const auto unselected = nlohmann::json::parse(ExecuteTool(
            nullptr, "patch_block_text", R"({"name":"note","field":"content","patch":"@@\n+x"})").content);
        CHECK(unselected["error"] == "No flowgraph is currently selected.");
    }

    REQUIRE(graph->blockDestroy("note", false) == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Text patches tolerate missing prefixes without guessing", "[agent][tools][text]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const std::string base = "def process(samples):\n    gain = 2.0\n    print(\"gain is\", gain)\n"
                             "    return samples * gain\n";
    const std::string halved = "def process(samples):\n    gain = 0.5\n    print(\"gain is\", gain)\n"
                               "    return samples * gain\n";

    const struct {
        std::string text;
        std::string patch;
        std::string expected;
        std::string note;
        std::string error;
        std::string absent;
    } cases[] = {
        {base, "def process(samples):\n-    gain = 2.0\n+    gain = 0.5", halved, "Patch line 1", "", ""},
        {base, "@@\ndef process(samples):\n-    gain = 2.0\n+    gain = 0.5\n    print(\"gain is\", gain)\n@@",
         halved, "Patch line 2", "", ""},
        {base, "@@\n    gain = 2.0\n+    print(\"x\")\n    print(\"gain is\", gain)",
         "def process(samples):\n    gain = 2.0\n    print(\"x\")\n    print(\"gain is\", gain)\n"
         "    return samples * gain\n", "", "", ""},
        {base, "@@\n-    gain = 2.0\n+    gain = 0.5\n@@\n", halved, "", "", ""},
        {base, "@@ def process(samples):\n-gain = 2.0\n+gain = 0.5", halved, "", "", ""},
        {base, "def process(samples):\n    return samples\n", base, "", "changes nothing", ""},
        {"def f():\n    value = 1\n    value = 1\n", "value = 1\n+    other = 2",
         "def f():\n    value = 1\n    value = 1\n", "", "matches 2 places", ""},
        {base, "@@\n     gain = 2.0\n-    print(\\\"gain is\\\", gain)\n+    print(\\\"g\\\")", base, "",
         "escaped one time too many", ""},
        {"s = \"a\\\"b\"\n", "@@\n-s = \"a\\\"c\"\n+s = 1", "s = \"a\\\"b\"\n", "", "Nothing was changed.",
         "escaped one time too many"},
        {base, "@@\n    gain = 2.0\n+   scale = 3\n    print(\"gain is\", gain)",
         "def process(samples):\n    gain = 2.0\n    scale = 3\n    print(\"gain is\", gain)\n"
         "    return samples * gain\n", "", "", ""},
        {base, "@@\n    gain = 2.0\n-   print(\"gain is\", gain)\n+   print(\"g\")",
         "def process(samples):\n    gain = 2.0\n    print(\"g\")\n    return samples * gain\n", "", "", ""},
        {"/*\n * first\n * second\n */\n", "@@\n * first\n+* inserted\n * second",
         "/*\n * first\n * inserted\n * second\n */\n", "", "", ""},
        {"Title\nIntro\n+ plus item\nTail\n", "@@\nTitle\n-Intro\n+Summary\n+ plus item",
         "Title\nIntro\n+ plus item\nTail\n", "", "is already in the text next to this hunk", ""},
    };

    for (const auto& test : cases) {
        CAPTURE(test.text, test.patch);

        const nlohmann::json create = {{"operations", nlohmann::json::array({
            {{"op", "create_block"}, {"type", "note"}, {"name", "note"}, {"config", {{"content", test.text}}}},
        })}};
        REQUIRE_FALSE(ExecuteTool(graph, "apply_flowgraph_patch", create.dump()).error);

        const nlohmann::json edit = {{"name", "note"}, {"field", "content"}, {"patch", test.patch}};
        const auto result = ExecuteTool(graph, "patch_block_text", edit.dump());
        const auto body = nlohmann::json::parse(result.content);

        CHECK(result.error == !test.error.empty());
        if (test.error.empty()) {
            CHECK(body.contains("note") == !test.note.empty());
            if (!test.note.empty()) {
                CHECK(body["note"].get<std::string>().find(test.note) != std::string::npos);
                CHECK(body["note"].get<std::string>().find("read as an unchanged line") != std::string::npos);
            }
        } else {
            const auto error = body["error"].get<std::string>();
            CHECK(error.find(test.error) != std::string::npos);
            if (!test.absent.empty()) {
                CHECK(error.find(test.absent) == std::string::npos);
            }
        }

        const auto note = nlohmann::json::parse(
            ExecuteTool(graph, "inspect_block", R"({"name":"note"})").content);
        CHECK(note["config"]["content"] == test.expected);

        REQUIRE(graph->blockDestroy("note", false) == Result::SUCCESS);
    }

    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Saved flowgraphs are listed and opened only from the user's folders", "[agent][tools][saved]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    const auto root = std::filesystem::temp_directory_path() /
                      ("cyberether-saved-flowgraphs-" + std::to_string(std::random_device{}()));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "mine" / "nested");
    std::filesystem::create_directories(root / "elsewhere");

    const auto write = [](const std::filesystem::path& path, const std::string& content) {
        std::ofstream(path) << content;
    };
    const std::string graph = "---\nversion: 2\ntitle: Cell Scan\nsummary: 'Watch the cellular bands.'\n"
                              "graph:\n  - name: scan_note\n    module: note\n    device: cpu\n"
                              "    runtime: native\n    config:\n      content: hello\n";
    write(root / "mine" / "Cell Ops", graph);
    write(root / "mine" / "second.yml", "---\nversion: 2\ngraph:\n  - name: other\n    module: note\n");
    write(root / "mine" / "notes.txt", graph);
    write(root / "mine" / "script", "#!/usr/bin/env python3\nprint('version: 2')\ngraph: no\n");
    write(root / "mine" / "typed", "version: str = \"2\"\ntitle: str = \"token\"\ngraph: dict = {}\n");
    write(root / "mine" / "config", "version: 1\nsummary: api_key=secret\ngraph: x\n  - name: password\n");
    write(root / "mine" / "nested" / "deep.yml", graph);
    write(root / "elsewhere" / "outside.yml", graph);

    GraphHost host(nullptr);
    const Agent::Tools tools(host);

    SECTION("nothing is listed until a folder is known") {
        const auto body = nlohmann::json::parse(tools.execute(nullptr, "list_flowgraphs", "{}").content);
        CHECK(body["count"] == 0);
        CHECK(body.contains("note"));
    }

    host.folders = {(root / "mine").string(), (root / "missing").string()};
    const auto listed = nlohmann::json::parse(tools.execute(nullptr, "list_flowgraphs", "{}").content);

    SECTION("only flowgraph documents directly inside known folders are listed") {
        REQUIRE(listed["count"] == 2);
        CHECK(listed["folders"].size() == 1);
        CHECK(listed["flowgraphs"][0]["name"] == "Cell Ops");
        CHECK(listed["flowgraphs"][0]["title"] == "Cell Scan");
        CHECK(listed["flowgraphs"][0]["summary"] == "Watch the cellular bands.");
        CHECK(listed["flowgraphs"][0]["block_count"] == 1);
        CHECK(listed["flowgraphs"][0]["blocks"] == nlohmann::json::array({"scan_note"}));
        CHECK(listed["flowgraphs"][1]["name"] == "second");
    }

    SECTION("a listed path opens in a new graph") {
        const nlohmann::json args = {{"path", listed["flowgraphs"][0]["path"]}};
        const auto result = tools.execute(nullptr, "open_flowgraph", args.dump());
        REQUIRE_FALSE(result.error);

        const auto body = nlohmann::json::parse(result.content);
        CHECK(body["name"] == "Cell Ops");
        CHECK(body["opened"] == "Cell_Ops");
        REQUIRE(host.createdGraphs.size() == 1);
        CHECK(host.createdGraphs.front()->view().has("scan_note"));
        CHECK(body["flowgraph"]["block_count"] == 1);
    }

    SECTION("files that only resemble a flowgraph are not listed") {
        for (const auto& entry : listed["flowgraphs"]) {
            CHECK(entry["name"] != "typed");
            CHECK(entry["name"] != "config");
            CHECK(entry["name"] != "script");
        }
    }

    SECTION("documents written by other tools are still recognised") {
        write(root / "mine" / "windows.YML", "\xEF\xBB\xBF---\r\nversion: 2\r\ngraph:\r\n  - name: a\r\n");
        write(root / "mine" / "described.yaml", "--- # exported\nauthor: me\ndescription: d\nversion: 2\n"
                                                "graph:\n- name: flat\n  module: note\n- module: note\n"
                                                "  name: later\nmeta:\n  - name: not_a_block\n");
        write(root / "mine" / "latin.yml", "---\nversion: 2\ntitle: caf\xe9\ngraph:\n  - name: b\xe9ta\n");
        write(root / "mine" / "caf\xe9.yml", "---\nversion: 2\ngraph:\n  - name: hidden\n");

        const auto result = tools.execute(nullptr, "list_flowgraphs", "{}");
        REQUIRE_FALSE(result.error);
        const auto again = nlohmann::json::parse(result.content);

        std::map<std::string, nlohmann::json> byName;
        for (const auto& entry : again["flowgraphs"]) {
            byName[entry["name"].get<std::string>()] = entry;
        }
        REQUIRE(byName.contains("windows"));
        CHECK(byName["windows"]["blocks"] == nlohmann::json::array({"a"}));
        REQUIRE(byName.contains("described"));
        CHECK(byName["described"]["block_count"] == 2);
        CHECK(byName["described"]["blocks"] == nlohmann::json::array({"flat"}));
        REQUIRE(byName.contains("latin"));
        CHECK(byName["latin"]["title"].get<std::string>().starts_with("caf"));
        CHECK(again["count"] == 5);
    }

    SECTION("a symlink cannot reach outside a known folder") {
        std::filesystem::create_symlink(root / "elsewhere" / "outside.yml", root / "mine" / "link.yml");
        std::filesystem::create_directory_symlink(root / "elsewhere", root / "mine" / "linked");

        const auto again = nlohmann::json::parse(tools.execute(nullptr, "list_flowgraphs", "{}").content);
        CHECK(again["count"] == 2);

        for (const auto& path : {root / "mine" / "link.yml", root / "mine" / "linked" / "outside.yml"}) {
            const nlohmann::json args = {{"path", path.string()}};
            CHECK(tools.execute(nullptr, "open_flowgraph", args.dump()).error);
        }
        CHECK(host.createdGraphs.empty());
    }

    SECTION("a file that fails to load is reported with a warning not to save") {
        write(root / "mine" / "broken.yml", "---\nversion: 2\ngraph:\n  - name: bad\n    module: no_such_module\n");
        const nlohmann::json args = {{"path", (root / "mine" / "broken.yml").string()}};
        const auto result = tools.execute(nullptr, "open_flowgraph", args.dump());

        CHECK(result.error);
        const auto error = nlohmann::json::parse(result.content)["error"].get<std::string>();
        CHECK(error.find("could not be loaded") != std::string::npos);
        CHECK(error.find("without saving") != std::string::npos);
        REQUIRE(host.createdGraphs.size() == 1);
    }

    SECTION("oversized or malformed paths are refused cheaply") {
        const nlohmann::json args = {{"path", std::string(20000, 'x')}};
        const auto result = tools.execute(nullptr, "open_flowgraph", args.dump());
        CHECK(result.error);
        CHECK(result.content.size() < 400);
        CHECK(host.createdGraphs.empty());
    }

    SECTION("paths outside the listing are refused before any graph is created") {
        for (const auto& path : {root / "elsewhere" / "outside.yml", root / "mine" / "nested" / "deep.yml",
                                 root / "mine" / "notes.txt", root / "mine" / ".." / "elsewhere" / "outside.yml"}) {
            CAPTURE(path);
            const nlohmann::json args = {{"path", path.string()}};
            const auto result = tools.execute(nullptr, "open_flowgraph", args.dump());
            CHECK(result.error);
            CHECK(nlohmann::json::parse(result.content)["error"].get<std::string>().find("list_flowgraphs") !=
                  std::string::npos);
        }
        CHECK(host.createdGraphs.empty());
        CHECK(tools.prepare(nullptr, "open_flowgraph", R"({"path":5})").output.has_value());
    }

    for (auto& created : host.createdGraphs) {
        created->destroy();
    }
    std::filesystem::remove_all(root);
}
