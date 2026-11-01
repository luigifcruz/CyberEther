#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <optional>
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

    explicit GraphHost(std::shared_ptr<Flowgraph> graph) : graph(std::move(graph)) {}

    bool has(const std::string&) override {
        return false;
    }

    Result create(const std::string&, std::shared_ptr<Flowgraph>&) override {
        return Result::ERROR;
    }
};

struct DocsHost : GraphHost {
    DocsHost() : GraphHost(nullptr) {}

    std::optional<std::vector<DocMatch>> searchDocs(const std::string& query, U64) override {
        if (query.empty()) {
            return std::vector<DocMatch>{{"remote", "Remote", {}, {}, "Share a session."}};
        }
        return std::vector<DocMatch>{{"remote", "Remote", "Pairing", "pairing", "Scan the code."}};
    }

    std::optional<DocPage> readDocs(const std::string& slug) override {
        if (slug != "remote") {
            return std::nullopt;
        }
        return DocPage{"remote", "Remote", "Share a session.", "# Remote"};
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
        "read_block_text",
        "find_connection_options",
        "validate_flowgraph",
        "inspect_environment",
        "inspect_block_view",
        "list_examples",
        "get_example",
        "open_example",
        "search_docs",
        "read_docs",
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
        }
    }

    REQUIRE(actual == expected);
}

TEST_CASE("Removed agent tools cannot be dispatched", "[agent][tools]") {
    const auto result = ExecuteTool(nullptr, "create_block", R"({"type":"signal_generator"})");
    const auto body = nlohmann::json::parse(result.content);

    REQUIRE(body["error"] == "Unknown tool 'create_block'.");
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

TEST_CASE("Flowgraph inspection reports only the visible canvas rectangle as the viewport",
          "[agent][tools][flowgraph]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto flowgraph = std::make_shared<Flowgraph>();
    REQUIRE(flowgraph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    GraphHost host(flowgraph);
    Agent::Tools tools(host);
    auto call = tools.prepare(flowgraph, "inspect_flowgraph", "{}");
    auto output = tools.run(call);
    REQUIRE_FALSE(output.error);

    Agent::Tools::Layout layout;
    layout.viewport = Agent::Tools::Layout::Viewport{.visible = {10.0f, 20.0f, 800.0f, 600.0f}};
    Agent::Tools::Annotate(call, output, layout);

    const auto body = nlohmann::json::parse(output.content);
    REQUIRE(body.contains("viewport"));
    const auto& viewport = body["viewport"];
    CHECK(viewport.size() == 1);
    REQUIRE(viewport.contains("visible"));
    CHECK_FALSE(viewport.contains("toolbar"));
    CHECK_FALSE(viewport.contains("open_area"));
    CHECK(viewport["visible"]["x"] == 10.0);
    CHECK(viewport["visible"]["y"] == 20.0);
    CHECK(viewport["visible"]["width"] == 800.0);
    CHECK(viewport["visible"]["height"] == 600.0);
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

TEST_CASE("Agent docs tools read through the host", "[agent][tools][docs]") {
    GraphHost bare(nullptr);
    const auto unavailable = nlohmann::json::parse(Agent::Tools(bare).execute(nullptr, "search_docs", "{}").content);
    REQUIRE(unavailable.contains("error"));

    DocsHost host;
    const Agent::Tools tools(host);

    const auto search = nlohmann::json::parse(tools.execute(nullptr, "search_docs", R"({"query":"pair"})").content);
    REQUIRE(search["matches"].size() == 1);
    CHECK(search["matches"][0]["page"] == "remote");
    CHECK(search["matches"][0]["section"] == "Pairing");

    const auto page = nlohmann::json::parse(tools.execute(nullptr, "read_docs", R"({"page":"remote"})").content);
    CHECK(page["body"] == "# Remote");

    const auto missing = nlohmann::json::parse(tools.execute(nullptr, "read_docs", R"({"page":"remot"})").content);
    REQUIRE(missing.contains("error"));
    CHECK(missing["error"].get<std::string>().find("remote") != std::string::npos);
}
