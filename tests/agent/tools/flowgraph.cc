#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <optional>
#include <set>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "jetstream/agent/tools.hh"
#include "jetstream/detail/block_impl.hh"
#include "jetstream/domains/core/python/block.hh"
#include "jetstream/logger.hh"
#include "jetstream/domains/core/slice/block.hh"
#include "jetstream/registry.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/runtime_context.hh"

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

struct ConfigMappingConfig : Blocks::Python {
    std::string type() const override {
        return "__ce_agent_config_mapping";
    }
};

struct ConfigMappingImpl : Block::Impl {
    Result define() override {
        JST_CHECK(defineInterfaceConfig("outputTensor0", "Output 0", "",
                                       {{"type", "tensor-config"}, {"source", "outputTensorSpecs"}}));
        return defineInterfaceConfig("code", "Code", "",
                                     {{"type", "text"}, {"source", "file"}, {"index", U64{7}}});
    }
};

struct BlockRegistrationGuard {
    std::string type;

    ~BlockRegistrationGuard() {
        (void)Registry::UnregisterBlock(type);
    }
};

struct HealthFixtureConfig : Block::Config {
    std::string code;

    JST_BLOCK_TYPE(agent_health_fixture);
    JST_BLOCK_DOMAIN("Test");
    JST_BLOCK_PARAMS(code);
    JST_BLOCK_DESCRIPTION("Health Fixture", "Reports a diagnostic.", "A test-only block with a long code field.");
};

struct HealthFixtureBlock : Block::Impl, DynamicConfig<HealthFixtureConfig> {
    Result define() override {
        JST_CHECK(defineInterfaceInput("in", "Input", "Input tensor."));
        JST_CHECK(defineInterfaceConfig("code", "Code", "Long code field.", {{"type", "text"}}));
        const Parser::Map format = {{"type", "python-diagnostic"}, {"visibility", "internal"}};
        return defineInterfaceMetric("health", "Health", "Console output.", format, []() -> std::any {
            return Runtime::Context::Diagnostic{false, "stalled", {"first line", "last line"}};
        });
    }
};

JST_REGISTER_BLOCK_NO_MODULES(HealthFixtureBlock);

std::string LongCode(const std::string& marker) {
    std::string code = "def compute(ctx):\n    ctx.outputs[0][...] = ctx.inputs[0]\n";
    while (code.size() < 8192) {
        code += "# " + marker + "\n";
    }
    return code;
}

nlohmann::json InspectBlockArgs(const std::string& name, const nlohmann::json& includeConfig) {
    return {{"name", name}, {"include_config", includeConfig}};
}

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
        "inspect_tensor",
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
        "read_log",
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

TEST_CASE("Block docs expose ports and config keys before an instance exists", "[agent][tools][catalog]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    const auto generator = nlohmann::json::parse(
        ExecuteTool(nullptr, "get_block_docs", R"({"type":"signal_generator"})").content);
    REQUIRE_FALSE(generator.contains("error"));
    REQUIRE(generator["interface_scope"] == "default_config");
    REQUIRE(generator["observed_interfaces"].empty());
    REQUIRE(generator["input_ports"].empty());
    REQUIRE(generator["output_ports"][0]["name"] == "signal");
    REQUIRE(generator["output_ports"][0]["label"] == "Output");

    const auto analyzer = nlohmann::json::parse(
        ExecuteTool(nullptr, "get_block_docs", R"({"type":"spectrum_analyzer"})").content);
    REQUIRE_FALSE(analyzer.contains("error"));
    REQUIRE(analyzer["input_ports"][0]["name"] == "buffer");
    const auto& configKeys = analyzer["config_keys"];
    REQUIRE(std::find(configKeys.begin(), configKeys.end(), "lineplotAveraging") != configKeys.end());
    REQUIRE(std::find(configKeys.begin(), configKeys.end(), "spectrumAveraging") == configKeys.end());
    const auto averaging = std::find_if(analyzer["config_fields"].begin(), analyzer["config_fields"].end(),
                                      [](const auto& field) { return field["name"] == "lineplotAveraging"; });
    REQUIRE(averaging != analyzer["config_fields"].end());
    REQUIRE((*averaging)["config_key"] == "lineplotAveraging");

    const auto python = nlohmann::json::parse(
        ExecuteTool(nullptr, "get_block_docs", R"({"type":"python"})").content);
    REQUIRE_FALSE(python.contains("error"));
    REQUIRE(python["input_ports"][0]["name"] == "input0");
    REQUIRE(python["output_ports"][0]["name"] == "output0");
    const auto output = std::find_if(python["config_fields"].begin(), python["config_fields"].end(),
                                   [](const auto& field) { return field["name"] == "outputTensor0"; });
    REQUIRE(output != python["config_fields"].end());
    REQUIRE((*output)["config_key"] == "outputTensorSpecs");
    REQUIRE((*output)["config_index"] == 0);
}

TEST_CASE("Describing a prototype does not create modules or tensors", "[agent][tools][catalog]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    for (const auto& type : {"signal_generator", "spectrum_analyzer", "python"}) {
        CAPTURE(type);
        std::shared_ptr<Block> block;
        REQUIRE(Registry::BuildBlock(type, block) == Result::SUCCESS);
        REQUIRE(block->describe() == Result::SUCCESS);
        REQUIRE(block->describe() == Result::SUCCESS);
        REQUIRE(block->state() == Block::State::None);
        REQUIRE(block->inputs().empty());
        REQUIRE(block->outputs().empty());
        REQUIRE(block->modules().empty());
        REQUIRE(block->surfaces().empty());
    }
}

TEST_CASE("Block docs retain observed interfaces for configured instances", "[agent][tools][catalog]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    const Parser::Map config = {{"inputCount", U64{2}}, {"source", std::string("file")}};
    REQUIRE(graph->blockCreate("python", "python", config, {}) == Result::SUCCESS);
    const auto docs = nlohmann::json::parse(
        ExecuteTool(graph, "get_block_docs", R"({"type":"python"})").content);
    REQUIRE_FALSE(docs.contains("error"));
    REQUIRE(docs["interface_scope"] == "observed");
    REQUIRE(docs["interface_block"] == "python");
    REQUIRE(docs["input_ports"].size() == 2);
    REQUIRE(docs["input_ports"][1]["name"] == "input1");
    REQUIRE(docs["observed_interfaces"][0]["input_ports"] == nlohmann::json::array({"input0", "input1"}));
    const auto console = std::find_if(docs["config_fields"].begin(), docs["config_fields"].end(),
                                    [](const auto& field) { return field["name"] == "console"; });
    REQUIRE(console != docs["config_fields"].end());
    REQUIRE_FALSE(console->contains("config_key"));
    for (const auto& field : docs["config_fields"]) {
        if (field.contains("config_key")) {
            REQUIRE(docs["default_config"].contains(field["config_key"].get<std::string>()));
        }
    }
    const auto inspected = nlohmann::json::parse(
        ExecuteTool(graph, "inspect_block", R"({"name":"python"})").content);
    const auto output = std::find_if(inspected["config_fields"].begin(), inspected["config_fields"].end(),
                                   [](const auto& field) { return field["name"] == "outputTensor0"; });
    REQUIRE(output != inspected["config_fields"].end());
    REQUIRE((*output)["config_key"] == "outputTensorSpecs");
    REQUIRE((*output)["config_index"] == 0);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Config discovery follows widget source and index contracts", "[agent][tools][catalog]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    const ConfigMappingConfig config;
    REQUIRE(Registry::RegisterBlock(config.type(), config.domain(), config.title(), config.summary(),
                                    config.description(), []() {
        return std::make_shared<Block>(std::make_shared<ConfigMappingImpl>(),
                                       std::make_shared<ConfigMappingConfig>(),
                                       std::make_shared<ConfigMappingConfig>());
    }) == Result::SUCCESS);
    const BlockRegistrationGuard cleanup{config.type()};
    const auto args = nlohmann::json({{"type", config.type()}}).dump();
    const auto docs = nlohmann::json::parse(ExecuteTool(nullptr, "get_block_docs", args).content);
    REQUIRE_FALSE(docs.contains("error"));
    REQUIRE(docs["config_fields"].size() == 2);
    REQUIRE(docs["config_fields"][0]["config_key"] == "outputTensorSpecs");
    REQUIRE(docs["config_fields"][0]["config_index"] == 0);
    REQUIRE(docs["config_fields"][1]["config_key"] == "code");
    REQUIRE_FALSE(docs["config_fields"][1].contains("config_index"));
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
                 "config":{"ratio":4}},
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

TEST_CASE("Flowgraph patches reject unknown config keys before creating blocks", "[agent][tools][validation]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const auto docs = nlohmann::json::parse(
        ExecuteTool(nullptr, "get_block_docs", R"({"type":"spectrum_analyzer"})").content);
    const std::vector<nlohmann::json> configs = {
        {{"spectrumAveraging", 8}},
        {{"lineplotAveragin", 8}, {"rangeMin", -80.0}},
        {{"spectrumAveraging", 8}, {"notASetting", true}},
        {{"spectrumAveraging", nullptr}},
        {{"LineplotAveraging", 8}},
        {{"", true}},
    };

    for (const auto& config : configs) {
        CAPTURE(config);

        const auto args = nlohmann::json({{"operations", {
            {{"op", "create_block"}, {"type", "spectrum_analyzer"}, {"name", "rejected"}, {"config", config}},
        }}}).dump();
        const auto result = ExecuteTool(graph, "apply_flowgraph_patch", args);
        const auto body = nlohmann::json::parse(result.content);

        CHECK(result.error);
        REQUIRE(body["status"] == "partial");
        REQUIRE(body["results"].size() == 1);
        const auto& failure = body["results"][0];
        REQUIRE(failure.contains("unknown_config_keys"));
        CHECK_FALSE(failure["unknown_config_keys"].empty());
        CHECK(failure["available_config_keys"] == docs["config_keys"]);
        CHECK_FALSE(failure.contains("may_have_applied"));
        for (const auto& [key, value] : config.items()) {
            if (!docs["default_config"].contains(key)) {
                CHECK(std::find(failure["unknown_config_keys"].begin(), failure["unknown_config_keys"].end(), key) !=
                      failure["unknown_config_keys"].end());
            }
        }
        if (config.contains("lineplotAveragin")) {
            CHECK(failure["error"].get<std::string>().find("lineplotAveraging") != std::string::npos);
        }
        CHECK(body["block_count"] == 0);
        CHECK_FALSE(graph->view().has("rejected"));
    }

    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Flowgraph patches reject unknown config keys without reconfiguring blocks", "[agent][tools][validation]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const auto created = nlohmann::json::parse(ExecuteTool(graph, "apply_flowgraph_patch", R"({
        "operations": [{"op":"create_block","type":"spectrum_analyzer","name":"analyzer",
                        "config":{"lineplotAveraging":4,"rangeMin":-90.0}}]
    })").content);
    REQUIRE(created["status"] == "success");
    const auto before = nlohmann::json::parse(
        ExecuteTool(graph, "inspect_block", R"({"name":"analyzer"})").content);

    const auto result = ExecuteTool(graph, "apply_flowgraph_patch", R"({
        "operations": [
            {"op":"reconfigure_block","name":"analyzer",
             "config":{"spectrumAveraging":8,"lineplotAveraging":16,"rangeMin":-40.0}},
            {"op":"create_block","type":"note","name":"must_not_run"}
        ]
    })");
    const auto body = nlohmann::json::parse(result.content);

    CHECK(result.error);
    REQUIRE(body["status"] == "partial");
    REQUIRE(body["results"].size() == 1);
    CHECK(body["results"][0]["unknown_config_keys"] == nlohmann::json::array({"spectrumAveraging"}));
    CHECK_FALSE(body["results"][0].contains("may_have_applied"));
    CHECK_FALSE(graph->view().has("must_not_run"));

    const auto after = nlohmann::json::parse(
        ExecuteTool(graph, "inspect_block", R"({"name":"analyzer"})").content);
    CHECK(after["config"] == before["config"]);
    CHECK(after["state"] == before["state"]);

    const auto changed = nlohmann::json::parse(ExecuteTool(graph, "apply_flowgraph_patch", R"({
        "operations": [{"op":"reconfigure_block","name":"analyzer","config":{"lineplotAveraging":8}}]
    })").content);
    REQUIRE(changed["status"] == "success");
    CHECK(changed["results"][0]["config"]["lineplotAveraging"] == 8);

    REQUIRE(graph->blockDestroy("analyzer", false) == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Config key validation preserves sequential edits and structured values", "[agent][tools][validation]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    const auto rejected = ExecuteTool(graph, "apply_flowgraph_patch", R"({
        "operations": [
            {"op":"create_block","type":"note","name":"retained","config":{}},
            {"op":"create_block","type":"python","name":"rejected",
             "config":{"outputTensor0":{"shape":"[2, 4]"}}}
        ]
    })");
    const auto body = nlohmann::json::parse(rejected.content);
    REQUIRE(rejected.error);
    REQUIRE(body["status"] == "partial");
    REQUIRE(body["results"].size() == 2);
    CHECK(body["results"][0]["status"] == "success");
    CHECK(body["results"][1]["unknown_config_keys"] == nlohmann::json::array({"outputTensor0"}));
    CHECK(graph->view().has("retained"));
    CHECK_FALSE(graph->view().has("rejected"));

    const auto accepted = nlohmann::json::parse(ExecuteTool(graph, "apply_flowgraph_patch", R"({
        "operations": [
            {"op":"create_block","type":"python","name":"python",
             "config":{"source":"file","outputTensorSpecs":[
                 {"shape":"[2, 4]","dtype":"CF32","device":"cpu","axes":"[C, S]"}
             ]}},
            {"op":"reconfigure_block","name":"python","config":{}}
        ]
    })").content);
    REQUIRE(accepted["status"] == "success");
    CHECK(accepted["results"][0]["config"]["outputTensorSpecs"][0]["dtype"] == "CF32");
    CHECK(accepted["results"][1]["config"].empty());

    REQUIRE(graph->blockDestroy("python", false) == Result::SUCCESS);
    REQUIRE(graph->blockDestroy("retained", false) == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
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

TEST_CASE("Agent log tool reads recent entries by severity", "[agent][tools][log]") {
    const int level = _JST_LOG_DEBUG_LEVEL();
    JST_LOG_SET_DEBUG_LEVEL(2);
    std::ostringstream output;
    std::ostream* previous = &JST_LOG_SINK();
    JST_LOG_SET_SINK(&output);
    JST_INFO("agent log probe info");
    JST_ERROR("agent log probe error");
    JST_ERROR("agent log detail header:\nsecond line detail");
    std::string accented;
    for (int i = 0; i < 3000; ++i) {
        accented += "\xC3\xA9";
    }
    JST_ERROR("agent log wide {}", accented);
    JST_ERROR("agent log escape before \x1b[2@after \x1b]0;title\x07" "done");
    JST_ERROR("agent log invalid \xFF byte");
    JST_LOG_SET_SINK(previous);
    JST_LOG_SET_DEBUG_LEVEL(level);
    CHECK(output.str().find("agent log probe error") != std::string::npos);

    const auto errors = nlohmann::json::parse(ExecuteTool(nullptr, "read_log", R"({"query":"agent log probe"})").content);
    REQUIRE(errors["entries"].size() == 1);
    CHECK(errors["entries"][0]["level"] == "error");
    CHECK(errors["entries"][0]["message"] == "agent log probe error");

    const auto all = nlohmann::json::parse(
        ExecuteTool(nullptr, "read_log", R"({"level":"info","query":"agent log probe"})").content);
    CHECK(all["entries"].size() == 2);

    const auto detail = nlohmann::json::parse(ExecuteTool(nullptr, "read_log", R"({"query":"second line"})").content);
    REQUIRE(detail["entries"].size() == 1);
    CHECK(detail["entries"][0]["message"] == "agent log detail header:\nsecond line detail");

    const auto wide = ExecuteTool(nullptr, "read_log", R"({"query":"agent log wide"})");
    const auto parsed = nlohmann::json::parse(wide.content, nullptr, false);
    REQUIRE_FALSE(parsed.is_discarded());
    REQUIRE(parsed["entries"].size() == 1);

    const auto escape = nlohmann::json::parse(ExecuteTool(nullptr, "read_log", R"({"query":"agent log escape"})").content);
    REQUIRE(escape["entries"].size() == 1);
    CHECK(escape["entries"][0]["message"] == "agent log escape before after done");

    const auto bytes = nlohmann::json::parse(
        ExecuteTool(nullptr, "read_log", R"({"query":"agent log invalid"})").content, nullptr, false);
    REQUIRE_FALSE(bytes.is_discarded());
    REQUIRE(bytes["entries"].size() == 1);

    const auto invalid = nlohmann::json::parse(ExecuteTool(nullptr, "read_log", R"({"level":"loud"})").content);
    CHECK(invalid.contains("error"));
}

TEST_CASE("Flowgraph inspection reports the editor selection", "[agent][tools][layout]") {
    Agent::Tools::Call call{.tool = "inspect_flowgraph"};
    for (const auto& definition : Agent::Tools::Definitions()) {
        if (definition.function.name == "inspect_flowgraph") {
            call.definition = &definition;
        }
    }
    REQUIRE(call.definition);

    Agent::Tools::Layout layout;
    Agent::Tools::Output output{R"({"blocks":[]})"};
    Agent::Tools::Annotate(call, output, layout);
    CHECK_FALSE(nlohmann::json::parse(output.content).contains("selected"));

    layout.selected = {"filter", "source"};
    Agent::Tools::Annotate(call, output, layout);
    const auto body = nlohmann::json::parse(output.content);
    CHECK(body["selected"] == nlohmann::json::array({"filter", "source"}));
}

TEST_CASE("Block inspection defines an optional config toggle", "[agent][tools][inspect]") {
    const auto& tools = Agent::Tools::Definitions();
    const auto tool = std::find_if(tools.begin(), tools.end(), [](const auto& definition) {
        return definition.function.name == "inspect_block";
    });
    REQUIRE(tool != tools.end());

    const auto schema = nlohmann::json::parse(tool->function.parametersJson);
    CHECK(schema["required"] == nlohmann::json::array({"name"}));
    CHECK(schema["properties"]["include_config"]["type"] == "boolean");
    CHECK(schema["additionalProperties"] == false);
}

TEST_CASE("Block inspection can omit only the config", "[agent][tools][inspect]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(graph->blockCreate("source", "signal_generator", {}, {}) == Result::SUCCESS);

    const std::string marker = "lightweight inspection marker";
    const auto code = LongCode(marker);
    REQUIRE(code.size() > 7000);

    TensorMap healthInputs;
    healthInputs["in"].requested("source", "signal");
    REQUIRE(graph->blockCreate("health", "agent_health_fixture", {{"code", code}}, healthInputs) == Result::SUCCESS);

    TensorMap pythonInputs;
    pythonInputs["input0"].requested("source", "signal");
    const Parser::Map pythonConfig = {{"code", code}};
    REQUIRE(graph->blockCreate("python", "python", pythonConfig, pythonInputs, DeviceType::CPU, RuntimeType::PYTHON) !=
            Result::FATAL);

    const Parser::Map brokenConfig = {{"source", std::string("invalid")}, {"code", code}};
    REQUIRE(graph->blockCreate("broken", "python", brokenConfig, {}, DeviceType::CPU, RuntimeType::PYTHON) !=
            Result::FATAL);

    const auto graphBefore = ExecuteTool(graph, "inspect_flowgraph", R"({"detail":"full"})").content;

    for (const auto* name : {"source", "health", "python", "broken"}) {
        CAPTURE(name);
        const auto baseline = ExecuteTool(graph, "inspect_block", nlohmann::json({{"name", name}}).dump());
        REQUIRE_FALSE(baseline.error);

        const auto explicitTrue = ExecuteTool(graph, "inspect_block", InspectBlockArgs(name, true).dump());
        CHECK_FALSE(explicitTrue.error);
        CHECK(explicitTrue.content == baseline.content);

        const auto omitted = ExecuteTool(graph, "inspect_block", InspectBlockArgs(name, false).dump());
        REQUIRE_FALSE(omitted.error);
        CHECK(omitted.content.find(R"("config":)") == std::string::npos);

        auto expected = nlohmann::json::parse(baseline.content);
        REQUIRE(expected.contains("config"));
        expected.erase("config");
        CHECK(nlohmann::json::parse(omitted.content) == expected);
    }

    SECTION("long code stays out of the response") {
        const auto args = InspectBlockArgs("python", false).dump();
        const auto full = ExecuteTool(graph, "inspect_block", R"({"name":"python"})");
        const auto omitted = ExecuteTool(graph, "inspect_block", args);

        CHECK(nlohmann::json::parse(full.content)["config"]["code"] == code);
        CHECK(omitted.content.find(marker) == std::string::npos);
        CHECK(omitted.content.size() + code.size() <= full.content.size());
    }

    SECTION("health fields survive without the config") {
        const auto health = nlohmann::json::parse(
            ExecuteTool(graph, "inspect_block", InspectBlockArgs("health", false).dump()).content);

        CHECK(health["inputs"]["in"] == nlohmann::json({{"block", "source"}, {"port", "signal"}}));
        CHECK(health["input_ports"][0]["name"] == "in");
        CHECK(health["config_fields"][0]["config_key"] == "code");
        CHECK(health["surfaces"] == nlohmann::json::array());
        CHECK(health["node"].contains("x"));
        CHECK(health["layout_status"] == "pending");
        CHECK(health["bounds"].is_null());

        REQUIRE(health["metrics"].size() == 1);
        CHECK(health["metrics"][0]["name"] == "health");
        CHECK(health["metrics"][0]["value"]["healthy"] == false);
        CHECK(health["metrics"][0]["value"]["status"] == "stalled");
        CHECK(health["metrics"][0]["value"]["console"] == nlohmann::json::array({"first line", "last line"}));

        const auto python = nlohmann::json::parse(
            ExecuteTool(graph, "inspect_block", InspectBlockArgs("python", false).dump()).content);
        CHECK(python["metrics"][0]["name"] == "pythonDiagnostic");
        CHECK(python["metrics"][0]["value"].contains("console"));

        const auto broken = nlohmann::json::parse(
            ExecuteTool(graph, "inspect_block", InspectBlockArgs("broken", false).dump()).content);
        CHECK(broken["state"] == "errored");
        CHECK_FALSE(broken["diagnostic"].get<std::string>().empty());
    }

    SECTION("measured layout is annotated without the config") {
        const auto args = InspectBlockArgs("python", false).dump();
        GraphHost host(graph);
        Agent::Tools tools(host);
        auto call = tools.prepare(graph, "inspect_block", args);
        REQUIRE(call.definition);

        auto output = tools.run(call);
        REQUIRE_FALSE(output.error);

        Agent::Tools::Layout layout;
        layout.nodes["python"] = {
            .bounds = {20.0f, 30.0f, 236.0f, 186.0f},
            .size = {220.0f, 170.0f},
            .content = Extent2D<F32>{300.0f, 100.0f},
        };
        Agent::Tools::Annotate(call, output, layout);

        const auto body = nlohmann::json::parse(output.content);
        CHECK_FALSE(body.contains("config"));
        CHECK(body["layout_status"] == "measured");
        CHECK(body["bounds"].is_object());
        CHECK(body["clipped"] == true);
    }

    CHECK(ExecuteTool(graph, "inspect_flowgraph", R"({"detail":"full"})").content == graphBefore);
    CHECK(nlohmann::json::parse(ExecuteTool(graph, "inspect_block", R"({"name":"health"})").content)["config"]["code"] == code);

    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Block detail profiles shape the tool responses", "[agent][tools][inspect]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(graph->blockCreate("source", "signal_generator", {}, {}) == Result::SUCCESS);

    const auto code = LongCode("detail profile marker");
    TensorMap inputs;
    inputs["in"].requested("source", "signal");
    REQUIRE(graph->blockCreate("health", "agent_health_fixture", {{"code", code}}, inputs) == Result::SUCCESS);

    const auto listed = [&](const std::string& detail) {
        const auto args = nlohmann::json({{"detail", detail}}).dump();
        const auto body = nlohmann::json::parse(ExecuteTool(graph, "inspect_flowgraph", args).content);
        const auto block = std::find_if(body["blocks"].begin(), body["blocks"].end(), [](const auto& entry) {
            return entry["name"] == "health";
        });
        REQUIRE(block != body["blocks"].end());
        return nlohmann::json(*block);
    };

    const auto summary = listed("summary");
    for (const auto* key : {"device", "runtime", "provider", "summary", "config_fields"}) {
        CAPTURE(key);
        CHECK_FALSE(summary.contains(key));
    }
    CHECK(summary["input_ports"] == nlohmann::json::array({"in"}));
    const auto compact = summary["config"]["code"].get<std::string>();
    CHECK(compact.starts_with(code.substr(0, 256)));
    CHECK(compact.size() < code.size());
    CHECK(compact.find("inspect_block shows all") != std::string::npos);

    const auto full = listed("full");
    for (const auto* key : {"device", "runtime", "provider", "summary"}) {
        CAPTURE(key);
        CHECK(full.contains(key));
    }
    CHECK(full["config"]["code"] == code);
    CHECK(full["config_fields"][0]["config_key"] == "code");
    CHECK(full["input_ports"][0]["name"] == "in");

    const auto health = nlohmann::json::parse(
        ExecuteTool(graph, "inspect_block", InspectBlockArgs("health", false).dump()).content);
    CHECK_FALSE(health.contains("config"));
    for (const auto& [key, value] : full.items()) {
        CAPTURE(key);
        if (key != "config") {
            CHECK(health[key] == value);
        }
    }

    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Block inspection rejects a malformed config toggle", "[agent][tools][inspect]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(graph->blockCreate("source", "signal_generator", {}, {}) == Result::SUCCESS);

    const std::vector<nlohmann::json> malformed = {
        nullptr, "false", "true", 0, 1, 0.5, nlohmann::json::array(), nlohmann::json::array({false}),
        nlohmann::json::object(), nlohmann::json({{"value", false}}),
    };
    for (const auto& value : malformed) {
        CAPTURE(value.dump());
        const auto output = ExecuteTool(graph, "inspect_block", InspectBlockArgs("source", value).dump());
        CHECK(output.error);
        CHECK(nlohmann::json::parse(output.content) ==
              nlohmann::json({{"error", "'include_config' must be a boolean."}}));
    }

    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Block inspection keeps lookup errors with the config toggle", "[agent][tools][inspect]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    const auto sameFailure = [](const std::shared_ptr<Flowgraph>& graph, const nlohmann::json& args) {
        const auto baseline = ExecuteTool(graph, "inspect_block", args.dump());
        REQUIRE(baseline.error);

        auto withFlag = args;
        withFlag["include_config"] = false;
        const auto omitted = ExecuteTool(graph, "inspect_block", withFlag.dump());
        CHECK(omitted.error);
        CHECK(omitted.content == baseline.content);
    };

    sameFailure(nullptr, {{"name", "source"}});

    const auto noGraph = ExecuteTool(nullptr, "inspect_block", InspectBlockArgs("source", "false").dump());
    CHECK(noGraph.error);
    CHECK(noGraph.content.find("No flowgraph") != std::string::npos);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(graph->blockCreate("source", "signal_generator", {}, {}) == Result::SUCCESS);

    sameFailure(graph, nlohmann::json::object());
    sameFailure(graph, {{"name", ""}});
    sameFailure(graph, {{"name", "absent"}});
    sameFailure(graph, {{"name", 7}});

    REQUIRE(graph->destroy() == Result::SUCCESS);
}
