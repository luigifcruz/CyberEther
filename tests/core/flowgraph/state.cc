#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "jetstream/detail/flowgraph_impl.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_environment.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/tensor_link.hh"
#include "flowgraph_fixture.hh"

namespace {

using namespace Jetstream;

class CreatedFlowgraph {
 public:
    CreatedFlowgraph() {
        REQUIRE(TestFlowgraph::RegisterSyntheticGraph() == Result::SUCCESS);
        TestFlowgraph::syntheticFaultState().reset();
        created = flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS;
        REQUIRE(created);
    }

    ~CreatedFlowgraph() {
        TestFlowgraph::syntheticFaultState().reset();
        if (created) {
            (void)flowgraph.destroy();
        }
    }

    Flowgraph flowgraph;

 private:
    bool created = false;
};

Result Import(Flowgraph& flowgraph, std::string_view yaml) {
    const std::vector<char> blob(yaml.begin(), yaml.end());
    return flowgraph.importFromBlob(blob);
}

Parser::Map StateValue(const std::string& value) {
    Parser::Map state;
    state["value"] = value;
    return state;
}

std::string StateValue(const Parser::Map& state) {
    return std::any_cast<std::string>(state.at("value"));
}

bool Contains(const std::vector<std::string>& values, const std::string& expected) {
    return std::find(values.begin(), values.end(), expected) != values.end();
}

bool Contains(const std::vector<Flowgraph::View::InterfaceEntry>& entries,
              const std::string& expected) {
    return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.name == expected;
    });
}

std::unordered_map<std::string, U64> VersionMap(
    const std::vector<std::pair<std::string, U64>>& versions) {
    return {versions.begin(), versions.end()};
}

void RequireRejectedWithoutBlocks(Flowgraph& flowgraph, std::string_view yaml) {
    REQUIRE(Import(flowgraph, yaml) == Result::ERROR);
    REQUIRE(flowgraph.view().empty());
}

}  // namespace

TEST_CASE("Flowgraph import rejects malformed documents and field types",
          "[core][flowgraph][state][import]") {
    CreatedFlowgraph test;
    auto& flowgraph = test.flowgraph;

    SECTION("invalid YAML") {
        RequireRejectedWithoutBlocks(flowgraph, "version: [2\n");
    }

    SECTION("non-map document") {
        RequireRejectedWithoutBlocks(flowgraph, "not-a-map\n");
    }

    SECTION("non-scalar version") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: [2]
graph: []
)");
    }

    SECTION("version two graph must be a sequence") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  source:
    module: flowgraph_test_source
)");
    }

    SECTION("graph entries must be maps") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - flowgraph_test_source
)");
    }

    SECTION("config must be a map") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: source
    module: flowgraph_test_source
    config: [1, 2]
)");
    }

    SECTION("input must be a map") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: source
    module: flowgraph_test_source
    input: source.signal
)");
    }

    SECTION("block name is required") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - module: flowgraph_test_source
)");
    }

    SECTION("block names cannot contain reference separators") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: source.bad
    module: flowgraph_test_source
)");
    }

    SECTION("module type is required") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: source
)");
    }
}

TEST_CASE("Flowgraph import reports non-string input references instead of throwing",
          "[core][flowgraph][state][import]") {
    CreatedFlowgraph test;
    auto& flowgraph = test.flowgraph;
    Result result = Result::SUCCESS;
    REQUIRE(flowgraph.setTitle("before") == Result::SUCCESS);

    REQUIRE_NOTHROW(result = Import(flowgraph, R"(---
version: 2
title: after
graph:
  - name: sink
    module: flowgraph_test_merge
    input:
      a:
        block: source
        port: signal
)"));
    REQUIRE(result == Result::ERROR);
    REQUIRE(flowgraph.view().empty());
    REQUIRE(flowgraph.title() == "before");
}

TEST_CASE("Flowgraph import validates references, uniqueness, dependencies, and acyclicity",
          "[core][flowgraph][state][import]") {
    CreatedFlowgraph test;
    auto& flowgraph = test.flowgraph;

    SECTION("malformed reference") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: source
    module: flowgraph_test_source
  - name: sink
    module: flowgraph_test_merge
    input:
      a: '${graph.source.signal}'
)");
    }

    SECTION("reference with an empty segment") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: sink
    module: flowgraph_test_merge
    input:
      a: '${graph..output.signal}'
)");
    }

    SECTION("duplicate names") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: source
    module: flowgraph_test_source
  - name: source
    module: flowgraph_test_source
)");
    }

    SECTION("missing dependency") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: sink
    module: flowgraph_test_merge
    input:
      a: '${graph.missing.output.signal}'
)");
    }

    SECTION("dependency cycle") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: first
    module: flowgraph_test_merge
    input:
      a: '${graph.second.output.sum}'
  - name: second
    module: flowgraph_test_merge
    input:
      a: '${graph.first.output.sum}'
)");
    }

    SECTION("self dependency") {
        RequireRejectedWithoutBlocks(flowgraph, R"(---
version: 2
graph:
  - name: recursive
    module: flowgraph_test_merge
    input:
      a: '${graph.recursive.output.sum}'
)");
    }
}

TEST_CASE("Flowgraph import failure is atomic",
          "[core][flowgraph][state][import][atomicity]") {
    CreatedFlowgraph test;
    auto& flowgraph = test.flowgraph;

    REQUIRE(flowgraph.blockCreate("seed", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);

    SECTION("document metadata is not applied before validation succeeds") {
        REQUIRE(flowgraph.setTitle("before") == Result::SUCCESS);
        REQUIRE(flowgraph.metadata().set("layout", StateValue("before")) == Result::SUCCESS);

        REQUIRE(Import(flowgraph, R"(---
version: 2
title: after
meta:
  layout:
    value: after
graph:
  - name: sink
    module: flowgraph_test_merge
    input:
      a: '${graph.missing.output.signal}'
)") == Result::ERROR);

        Parser::Map layout;
        REQUIRE(flowgraph.metadata().get("layout", layout) == Result::SUCCESS);

        CHECK(flowgraph.title() == "before");
        CHECK(StateValue(layout) == "before");
        REQUIRE(flowgraph.view().size() == 1);
        REQUIRE(flowgraph.view().has("seed"));
    }

    SECTION("blocks created before a later creation error are rolled back") {
        const auto seedOutputId = ViewBlock(flowgraph, "seed").outputs.at("signal").tensor.id();
        REQUIRE(Import(flowgraph, R"(---
version: 2
graph:
  - name: imported
    module: flowgraph_test_source
    meta:
      marker:
        value: imported
  - name: broken
    module: definitely_missing_block_type
    meta:
      marker:
        value: broken
)") == Result::ERROR);

        CHECK(flowgraph.view().size() == 1);
        CHECK_FALSE(flowgraph.view().has("imported"));
        REQUIRE(flowgraph.view().has("seed"));
        REQUIRE(ViewBlock(flowgraph, "seed").outputs.at("signal").tensor.id() == seedOutputId);
        CHECK_FALSE(flowgraph.metadata().has("marker", "imported"));
        CHECK_FALSE(flowgraph.metadata().has("marker", "broken"));
    }
}

TEST_CASE("Flowgraph environment tracks epochs, versions, ranges, and replacements",
          "[core][flowgraph][state][environment]") {
    CreatedFlowgraph test;
    auto& environment = test.flowgraph.environment();

    REQUIRE(environment.epoch() == 0);
    REQUIRE(environment.set("mode", StateValue("first"), 10, 20) == Result::SUCCESS);
    REQUIRE(environment.epoch() == 1);
    REQUIRE_FALSE(environment.has("mode", 9));
    REQUIRE(environment.has("mode", 10));
    REQUIRE(environment.has("mode", 20));
    REQUIRE_FALSE(environment.has("mode", 21));

    Parser::Map state;
    REQUIRE(environment.get("mode", state, 10) == Result::SUCCESS);
    REQUIRE(StateValue(state) == "first");

    REQUIRE(environment.set("mode", StateValue("replacement"), 10, 20) == Result::SUCCESS);
    REQUIRE(environment.epoch() == 2);
    state.clear();
    REQUIRE(environment.get("mode", state, 20) == Result::SUCCESS);
    REQUIRE(StateValue(state) == "replacement");

    std::vector<std::pair<std::string, U64>> versions;
    REQUIRE(environment.versions(versions) == Result::SUCCESS);
    auto versionMap = VersionMap(versions);
    REQUIRE(versionMap.size() == 1);
    REQUIRE(versionMap.at("mode") == 2);

    REQUIRE(environment.set("mode", StateValue("overlap"), 15, 30) == Result::SUCCESS);
    REQUIRE(environment.epoch() == 3);
    state.clear();
    REQUIRE(environment.get("mode", state, 14) == Result::SUCCESS);
    REQUIRE(StateValue(state) == "replacement");
    state.clear();
    REQUIRE(environment.get("mode", state, 15) == Result::SUCCESS);
    REQUIRE(StateValue(state) == "overlap");
    state.clear();
    REQUIRE(environment.get("mode", state, 30) == Result::SUCCESS);
    REQUIRE(StateValue(state) == "overlap");
    REQUIRE_FALSE(environment.has("mode", 31));

    REQUIRE(environment.set("invalid", StateValue("bad"), 40, 39) == Result::ERROR);
    REQUIRE(environment.epoch() == 3);
    REQUIRE_FALSE(environment.has("invalid", 40));
}

TEST_CASE("Flowgraph environment clear operations preserve version meaning",
          "[core][flowgraph][state][environment]") {
    CreatedFlowgraph test;
    auto& environment = test.flowgraph.environment();

    REQUIRE(environment.set("mode", StateValue("scan")) == Result::SUCCESS);
    REQUIRE(environment.set("rate", StateValue("48000")) == Result::SUCCESS);
    REQUIRE(environment.epoch() == 2);

    std::vector<std::string> keys;
    REQUIRE(environment.keys(keys) == Result::SUCCESS);
    REQUIRE(keys.size() == 2);
    REQUIRE(Contains(keys, "mode"));
    REQUIRE(Contains(keys, "rate"));

    REQUIRE(environment.clear("mode") == Result::SUCCESS);
    REQUIRE(environment.epoch() == 3);
    REQUIRE_FALSE(environment.has("mode"));
    REQUIRE(environment.has("rate"));

    std::vector<std::pair<std::string, U64>> versions;
    REQUIRE(environment.versions(versions) == Result::SUCCESS);
    const auto versionMap = VersionMap(versions);
    REQUIRE(versionMap.size() == 1);
    REQUIRE(versionMap.at("rate") == 2);

    REQUIRE(environment.clearAll() == Result::SUCCESS);
    REQUIRE(environment.epoch() == 4);
    REQUIRE_FALSE(environment.has("rate"));
    REQUIRE(environment.keys(keys) == Result::SUCCESS);
    REQUIRE(keys.empty());
    REQUIRE(environment.versions(versions) == Result::SUCCESS);
    REQUIRE(versions.empty());
}

TEST_CASE("Flowgraph metadata keeps graph and block scopes independent",
          "[core][flowgraph][state][metadata]") {
    CreatedFlowgraph test;
    auto& flowgraph = test.flowgraph;
    auto& metadata = flowgraph.metadata();

    REQUIRE(flowgraph.blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph.blockCreate("other", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(metadata.set("layout", StateValue("graph")) == Result::SUCCESS);
    REQUIRE(metadata.set("layout", StateValue("source"), "source") == Result::SUCCESS);
    REQUIRE(metadata.set("layout", StateValue("other"), "other") == Result::SUCCESS);

    REQUIRE(metadata.has("layout"));
    REQUIRE(metadata.has("layout", "source"));
    REQUIRE(metadata.has("layout", "other"));

    Parser::Map state;
    REQUIRE(metadata.get("layout", state) == Result::SUCCESS);
    REQUIRE(StateValue(state) == "graph");
    state.clear();
    REQUIRE(metadata.get("layout", state, "source") == Result::SUCCESS);
    REQUIRE(StateValue(state) == "source");
    state.clear();
    REQUIRE(metadata.get("layout", state, "other") == Result::SUCCESS);
    REQUIRE(StateValue(state) == "other");

    std::vector<std::string> keys;
    REQUIRE(metadata.keys(keys) == Result::SUCCESS);
    REQUIRE(keys.size() == 1);
    REQUIRE(Contains(keys, "layout"));
    REQUIRE(metadata.keys(keys, "source") == Result::SUCCESS);
    REQUIRE(keys.size() == 1);
    REQUIRE(Contains(keys, "layout"));

    REQUIRE(metadata.clear("layout", "source") == Result::SUCCESS);
    REQUIRE_FALSE(metadata.has("layout", "source"));
    REQUIRE(metadata.has("layout"));
    REQUIRE(metadata.has("layout", "other"));

    REQUIRE(metadata.clearAll() == Result::SUCCESS);
    REQUIRE_FALSE(metadata.has("layout"));
    REQUIRE_FALSE(metadata.has("layout", "other"));
}

TEST_CASE("Flowgraph view exposes consistent block state through every accessor",
          "[core][flowgraph][state][view]") {
    CreatedFlowgraph test;
    auto& flowgraph = test.flowgraph;
    auto& view = flowgraph.view();

    REQUIRE(view.empty());
    REQUIRE(flowgraph.blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE_FALSE(view.empty());
    REQUIRE(view.size() == 1);
    REQUIRE(view.has("source"));

    std::vector<std::string> keys;
    REQUIRE(view.keys(keys) == Result::SUCCESS);
    REQUIRE(keys.size() == 1);
    REQUIRE(keys.front() == "source");

    Flowgraph::View::BlockInfo info;
    REQUIRE(view.info("source", info) == Result::SUCCESS);
    REQUIRE(info.name == "source");
    REQUIRE(info.type == TestFlowgraph::kSyntheticSourceType);
    REQUIRE(info.title == "Synthetic Source");
    REQUIRE(info.summary == "Produces a test-owned tensor.");
    REQUIRE(info.description == "Flowgraph fixture source block.");
    REQUIRE(info.device == DeviceType::CPU);
    REQUIRE(info.runtime == RuntimeType::NATIVE);
    REQUIRE(info.provider == "generic");
    REQUIRE(info.state == Block::State::Created);
    REQUIRE(info.nodeSize == Block::NodeSize::S);
    REQUIRE(info.diagnostic.empty());

    Parser::Map config;
    REQUIRE(view.config("source", config) == Result::SUCCESS);
    REQUIRE(config.contains("bufferSize"));

    TensorMap inputs;
    REQUIRE(view.inputs("source", inputs) == Result::SUCCESS);
    REQUIRE(inputs.empty());

    TensorMap outputs;
    REQUIRE(view.outputs("source", outputs) == Result::SUCCESS);
    REQUIRE(outputs.size() == 1);
    REQUIRE(outputs.contains("signal"));
    REQUIRE(outputs.at("signal").resolved());
    REQUIRE(outputs.at("signal").external.has_value());
    REQUIRE(outputs.at("signal").external->block == "source");
    REQUIRE(outputs.at("signal").external->port == "signal");

    std::vector<Flowgraph::View::InterfaceEntry> interfaceInputs;
    std::vector<Flowgraph::View::InterfaceEntry> interfaceOutputs;
    std::vector<Flowgraph::View::InterfaceEntry> interfaceConfigs;
    REQUIRE(view.interfaceInputs("source", interfaceInputs) == Result::SUCCESS);
    REQUIRE(view.interfaceOutputs("source", interfaceOutputs) == Result::SUCCESS);
    REQUIRE(view.interfaceConfigs("source", interfaceConfigs) == Result::SUCCESS);
    REQUIRE(interfaceInputs.empty());
    REQUIRE(interfaceOutputs.size() == 1);
    REQUIRE(Contains(interfaceOutputs, "signal"));
    REQUIRE(Contains(interfaceConfigs, "bufferSize"));

    std::vector<Flowgraph::View::MetricEntry> metrics;
    std::vector<std::shared_ptr<Module::Surface>> surfaces;
    REQUIRE(view.metrics("source", metrics) == Result::SUCCESS);
    REQUIRE(view.surfaces("source", surfaces) == Result::SUCCESS);
    REQUIRE(metrics.size() == 1);
    REQUIRE(metrics.front().name == "runtime:source");
    REQUIRE(metrics.front().label == "source");
    REQUIRE(metrics.front().format == "private-timing");
    REQUIRE(metrics.front().help == "Runtime timing collected by the scheduler.");
    REQUIRE(std::any_cast<Module::Timing>(&metrics.front().value) != nullptr);
    REQUIRE(surfaces.empty());

    Flowgraph::View::BlockData block;
    REQUIRE(view.block("source", block) == Result::SUCCESS);
    REQUIRE(block.name == info.name);
    REQUIRE(block.type == info.type);
    REQUIRE(block.title == info.title);
    REQUIRE(block.summary == info.summary);
    REQUIRE(block.description == info.description);
    REQUIRE(block.device == info.device);
    REQUIRE(block.runtime == info.runtime);
    REQUIRE(block.provider == info.provider);
    REQUIRE(block.state == info.state);
    REQUIRE(block.nodeSize == info.nodeSize);
    REQUIRE(block.diagnostic == info.diagnostic);
    REQUIRE(block.config.size() == 2);
    REQUIRE(std::any_cast<U64>(block.config.at("bufferSize")) == 8192);
    REQUIRE(std::any_cast<F32>(block.config.at("value")) == 1.0f);
    REQUIRE(block.inputs.empty());
    REQUIRE(block.outputs.size() == 1);
    const auto& signal = block.outputs.at("signal");
    REQUIRE(signal.resolved());
    REQUIRE(signal.external.has_value());
    REQUIRE(signal.external->block == "source");
    REQUIRE(signal.external->port == "signal");
    REQUIRE(signal.producer.has_value());
    REQUIRE(signal.producer->module == "source-source");
    REQUIRE(signal.producer->port == "signal");
    REQUIRE(signal.tensor.device() == DeviceType::CPU);
    REQUIRE(signal.tensor.dtype() == DataType::F32);
    REQUIRE(signal.tensor.shape() == Shape{8192});
    REQUIRE(block.interfaceInputs.empty());
    REQUIRE(block.interfaceOutputs.size() == 1);
    REQUIRE(block.interfaceOutputs[0].name == "signal");
    REQUIRE(block.interfaceOutputs[0].label == "Output");
    REQUIRE(block.interfaceOutputs[0].format.empty());
    REQUIRE(block.interfaceOutputs[0].help == "Synthetic output tensor.");
    REQUIRE(block.interfaceConfigs.size() == 2);
    REQUIRE(block.interfaceConfigs[0].name == "bufferSize");
    REQUIRE(block.interfaceConfigs[0].label == "Buffer Size");
    REQUIRE(block.interfaceConfigs[0].format == "int");
    REQUIRE(block.interfaceConfigs[0].help == "Output tensor size.");
    REQUIRE(block.interfaceConfigs[1].name == "value");
    REQUIRE(block.interfaceConfigs[1].label == "Value");
    REQUIRE(block.interfaceConfigs[1].format == "float");
    REQUIRE(block.interfaceConfigs[1].help == "Output tensor value.");
    CHECK((block.metrics.size() == 1 &&
           block.metrics[0].name == "runtime:source" &&
           block.metrics[0].label == "source" &&
           block.metrics[0].format == "private-timing" &&
           block.metrics[0].help == "Runtime timing collected by the scheduler." &&
           std::any_cast<Module::Timing>(&block.metrics[0].value) != nullptr));
    REQUIRE(block.surfaces.empty());

    REQUIRE(view.info("missing", info) == Result::ERROR);
    REQUIRE(view.config("missing", config) == Result::ERROR);
    REQUIRE(view.inputs("missing", inputs) == Result::ERROR);
    REQUIRE(view.outputs("missing", outputs) == Result::ERROR);
    REQUIRE(view.interfaceInputs("missing", interfaceInputs) == Result::ERROR);
    REQUIRE(view.interfaceOutputs("missing", interfaceOutputs) == Result::ERROR);
    REQUIRE(view.interfaceConfigs("missing", interfaceConfigs) == Result::ERROR);
    REQUIRE(view.metrics("missing", metrics) == Result::ERROR);
    REQUIRE(view.surfaces("missing", surfaces) == Result::ERROR);
    REQUIRE(view.block("missing", block) == Result::ERROR);
}

TEST_CASE("Flowgraph state helpers fail safely after their weak backing expires",
          "[core][flowgraph][state][detached]") {
    auto backing = std::make_shared<Flowgraph::Impl>();
    Flowgraph::Metadata metadata(backing);
    Flowgraph::Environment environment(backing);
    Flowgraph::View view(backing);
    backing.reset();

    Parser::Map state;
    std::vector<std::string> keys;
    std::vector<std::pair<std::string, U64>> versions;
    REQUIRE_FALSE(metadata.has("key"));
    REQUIRE(metadata.get("key", state) == Result::ERROR);
    REQUIRE(metadata.keys(keys) == Result::ERROR);
    REQUIRE(metadata.set("key", state) == Result::ERROR);
    REQUIRE(metadata.clear("key") == Result::ERROR);
    REQUIRE(metadata.clearAll() == Result::ERROR);

    REQUIRE_FALSE(environment.has("key"));
    REQUIRE(environment.epoch() == 0);
    REQUIRE(environment.get("key", state) == Result::ERROR);
    REQUIRE(environment.keys(keys) == Result::ERROR);
    REQUIRE(environment.versions(versions) == Result::ERROR);
    REQUIRE(environment.set("key", state) == Result::ERROR);
    REQUIRE(environment.clear("key") == Result::ERROR);
    REQUIRE(environment.clearAll() == Result::ERROR);

    REQUIRE_FALSE(view.has("block"));
    REQUIRE(view.empty());
    REQUIRE(view.size() == 0);
    REQUIRE(view.keys(keys) == Result::ERROR);

    Flowgraph::View::BlockInfo info;
    Flowgraph::View::BlockData block;
    Parser::Map config;
    TensorMap inputs;
    TensorMap outputs;
    std::vector<Flowgraph::View::InterfaceEntry> interfaceEntries;
    std::vector<Flowgraph::View::MetricEntry> metrics;
    std::vector<std::shared_ptr<Module::Surface>> surfaces;
    REQUIRE(view.info("block", info) == Result::ERROR);
    REQUIRE(view.config("block", config) == Result::ERROR);
    REQUIRE(view.inputs("block", inputs) == Result::ERROR);
    REQUIRE(view.outputs("block", outputs) == Result::ERROR);
    REQUIRE(view.interfaceInputs("block", interfaceEntries) == Result::ERROR);
    REQUIRE(view.interfaceOutputs("block", interfaceEntries) == Result::ERROR);
    REQUIRE(view.interfaceConfigs("block", interfaceEntries) == Result::ERROR);
    REQUIRE(view.metrics("block", metrics) == Result::ERROR);
    REQUIRE(view.surfaces("block", surfaces) == Result::ERROR);
    REQUIRE(view.block("block", block) == Result::ERROR);
}

TEST_CASE("TensorLink transitions retain only the active endpoint state",
          "[core][flowgraph][state][tensor-link]") {
    Tensor tensor;
    REQUIRE(tensor.create(DeviceType::CPU, DataType::F32, {1}) == Result::SUCCESS);

    TensorLink link;
    REQUIRE_FALSE(link.producer.has_value());
    REQUIRE_FALSE(link.external.has_value());
    REQUIRE(link.tensor.empty());
    REQUIRE_FALSE(link.resolved());

    link.requested("source", "signal");
    REQUIRE(link.external.has_value());
    REQUIRE(link.external->block == "source");
    REQUIRE(link.external->port == "signal");
    REQUIRE_FALSE(link.producer.has_value());
    REQUIRE(link.tensor.empty());
    REQUIRE_FALSE(link.resolved());

    link.produced("module", "output", tensor);
    REQUIRE(link.producer.has_value());
    REQUIRE(link.producer->module == "module");
    REQUIRE(link.producer->port == "output");
    REQUIRE_FALSE(link.external.has_value());
    REQUIRE(link.tensor.id() == tensor.id());
    REQUIRE(link.resolved());

    link.exposedAs("block", "port");
    REQUIRE(link.external.has_value());
    REQUIRE(link.external->block == "block");
    REQUIRE(link.external->port == "port");
    REQUIRE(link.producer.has_value());
    REQUIRE(link.tensor.id() == tensor.id());
    REQUIRE(link.resolved());

    link.requested("other", "input");
    REQUIRE_FALSE(link.producer.has_value());
    REQUIRE(link.external.has_value());
    REQUIRE(link.external->block == "other");
    REQUIRE(link.tensor.empty());
    REQUIRE_FALSE(link.resolved());
}

TEST_CASE("TensorLink resolution requires a complete producer and tensor",
          "[core][flowgraph][state][tensor-link]") {
    Tensor tensor;
    REQUIRE(tensor.create(DeviceType::CPU, DataType::F32, {1}) == Result::SUCCESS);

    TensorLink link;
    link.produced("", "output", tensor);
    REQUIRE_FALSE(link.resolved());
    link.produced("module", "", tensor);
    REQUIRE_FALSE(link.resolved());
    link.produced("module", "output", Tensor{});
    REQUIRE_FALSE(link.resolved());
    link.produced("module", "output", tensor);
    REQUIRE(link.resolved());
}

TEST_CASE("Flowgraph export omits transient and empty state and round-trips graph state",
          "[core][flowgraph][state][export]") {
    CreatedFlowgraph source;
    auto& flowgraph = source.flowgraph;

    REQUIRE(flowgraph.blockCreate("left", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph.blockCreate("right", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    TensorMap addInputs;
    addInputs["a"].requested("left", "signal");
    addInputs["b"].requested("right", "signal");
    REQUIRE(flowgraph.blockCreate("sum", TestFlowgraph::kSyntheticMergeType, {}, addInputs) == Result::SUCCESS);
    REQUIRE(flowgraph.setSummary("round trip") == Result::SUCCESS);

    const Parser::Map emptyState;
    REQUIRE(flowgraph.metadata().set("layout", StateValue("graph")) == Result::SUCCESS);
    REQUIRE(flowgraph.metadata().set("empty", emptyState) == Result::SUCCESS);
    REQUIRE(flowgraph.metadata().set("layout", StateValue("block"), "sum") == Result::SUCCESS);
    REQUIRE(flowgraph.metadata().set("empty", emptyState, "sum") == Result::SUCCESS);
    REQUIRE(flowgraph.environment().set("session", StateValue("transient")) == Result::SUCCESS);

    std::vector<char> blob;
    REQUIRE(flowgraph.exportToBlob(blob) == Result::SUCCESS);

    CreatedFlowgraph restored;
    REQUIRE(restored.flowgraph.setTitle("preserved because source title is omitted") == Result::SUCCESS);
    REQUIRE(restored.flowgraph.importFromBlob(blob) == Result::SUCCESS);

    REQUIRE(restored.flowgraph.title() == "preserved because source title is omitted");
    REQUIRE(restored.flowgraph.summary() == "round trip");
    REQUIRE(restored.flowgraph.view().size() == 3);
    REQUIRE_FALSE(restored.flowgraph.environment().has("session"));

    Parser::Map state;
    REQUIRE(restored.flowgraph.metadata().get("layout", state) == Result::SUCCESS);
    REQUIRE(StateValue(state) == "graph");
    REQUIRE_FALSE(restored.flowgraph.metadata().has("empty"));
    state.clear();
    REQUIRE(restored.flowgraph.metadata().get("layout", state, "sum") == Result::SUCCESS);
    REQUIRE(StateValue(state) == "block");
    REQUIRE_FALSE(restored.flowgraph.metadata().has("empty", "sum"));

    std::vector<std::string> keys;
    REQUIRE(restored.flowgraph.view().keys(keys) == Result::SUCCESS);
    REQUIRE(keys.size() == 3);
    REQUIRE(keys[0] == "left");
    REQUIRE(keys[1] == "right");
    REQUIRE(keys[2] == "sum");

    Flowgraph::View::BlockData sum;
    REQUIRE(restored.flowgraph.view().block("sum", sum) == Result::SUCCESS);
    REQUIRE(sum.state == Block::State::Created);
    REQUIRE(sum.inputs.size() == 2);
    REQUIRE(sum.inputs.at("a").external.has_value());
    REQUIRE(sum.inputs.at("a").external->block == "left");
    REQUIRE(sum.inputs.at("a").external->port == "signal");
    REQUIRE(sum.inputs.at("b").external.has_value());
    REQUIRE(sum.inputs.at("b").external->block == "right");
    REQUIRE(sum.inputs.at("b").external->port == "signal");
    REQUIRE(sum.outputs.at("sum").resolved());

    Parser::Map config;
    REQUIRE(restored.flowgraph.view().config("left", config) == Result::SUCCESS);
    REQUIRE(config.contains("bufferSize"));
    REQUIRE(std::any_cast<U64>(config.at("bufferSize")) == 8192);
}

TEST_CASE("Flowgraph serialization preserves a non-default module provider",
          "[core][flowgraph][state][serialization][provider]") {
    CreatedFlowgraph source;
    auto& flowgraph = source.flowgraph;

    REQUIRE(flowgraph.blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    const auto config = ViewBlock(flowgraph, "source").config;
    REQUIRE(flowgraph.blockRecreate("source",
                                    config,
                                    DeviceType::CPU,
                                    RuntimeType::NATIVE,
                                    TestFlowgraph::kSyntheticSourceTestProvider) == Result::SUCCESS);
    REQUIRE(ViewBlock(flowgraph, "source").provider == TestFlowgraph::kSyntheticSourceTestProvider);

    std::vector<char> blob;
    REQUIRE(flowgraph.exportToBlob(blob) == Result::SUCCESS);
    const std::string yaml(blob.begin(), blob.end());
    REQUIRE(yaml.find("provider: flowgraph-test-alt") != std::string::npos);

    CreatedFlowgraph restored;
    REQUIRE(restored.flowgraph.importFromBlob(blob) == Result::SUCCESS);
    const auto block = ViewBlock(restored.flowgraph, "source");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.device == DeviceType::CPU);
    REQUIRE(block.runtime == RuntimeType::NATIVE);
    REQUIRE(block.provider == TestFlowgraph::kSyntheticSourceTestProvider);
}
