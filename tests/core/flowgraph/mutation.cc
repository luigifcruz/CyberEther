#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/flowgraph_metadata.hh"

namespace {

using namespace Jetstream;

TensorMap RequestedInput(const std::string& input,
                         const std::string& source,
                         const std::string& output) {
    TensorMap inputs;
    inputs[input].requested(source, output);
    return inputs;
}

std::vector<std::string> GraphKeys(Flowgraph& flowgraph) {
    std::vector<std::string> keys;
    REQUIRE(flowgraph.view().keys(keys) == Result::SUCCESS);
    return keys;
}

void CreateChain(Flowgraph& flowgraph) {
    REQUIRE(flowgraph.blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph.blockCreate("middle", TestFlowgraph::kSyntheticPassType, {},
                                  RequestedInput("buffer", "source", "signal")) == Result::SUCCESS);
    REQUIRE(flowgraph.blockCreate("leaf", TestFlowgraph::kSyntheticPassType, {},
                                  RequestedInput("buffer", "middle", "buffer")) == Result::SUCCESS);
}

void RequireState(Flowgraph& flowgraph, const std::string& block, Block::State state) {
    REQUIRE(ViewBlock(flowgraph, block).state == state);
}

void RequireConnection(Flowgraph& flowgraph,
                       const std::string& block,
                       const std::string& input,
                       const std::string& source,
                       const std::string& output) {
    const auto data = ViewBlock(flowgraph, block);
    REQUIRE(data.inputs.contains(input));
    REQUIRE(data.inputs.at(input).external.has_value());
    REQUIRE(data.inputs.at(input).external->block == source);
    REQUIRE(data.inputs.at(input).external->port == output);
}

void RequireNoConnection(Flowgraph& flowgraph,
                         const std::string& block,
                         const std::string& input) {
    REQUIRE_FALSE(ViewBlock(flowgraph, block).inputs.contains(input));
}

void SetTag(Flowgraph& flowgraph, const std::string& block, const std::string& value) {
    Parser::Map tag;
    tag["value"] = value;
    REQUIRE(flowgraph.metadata().set("tag", tag, block) == Result::SUCCESS);
}

void RequireTag(Flowgraph& flowgraph, const std::string& block, const std::string& value) {
    Parser::Map tag;
    REQUIRE(flowgraph.metadata().get("tag", tag, block) == Result::SUCCESS);
    REQUIRE(tag.contains("value"));
    REQUIRE(std::any_cast<std::string>(tag.at("value")) == value);
}

void RequireIntactChain(Flowgraph& flowgraph) {
    const std::vector<std::string> expected = {"source", "middle", "leaf"};
    REQUIRE(GraphKeys(flowgraph) == expected);
    RequireState(flowgraph, "source", Block::State::Created);
    RequireState(flowgraph, "middle", Block::State::Created);
    RequireState(flowgraph, "leaf", Block::State::Created);
    RequireConnection(flowgraph, "middle", "buffer", "source", "signal");
    RequireConnection(flowgraph, "leaf", "buffer", "middle", "buffer");
}

bool IsCreated(Flowgraph& flowgraph, const std::string& block) {
    Flowgraph::View::BlockData data;
    return flowgraph.view().block(block, data) == Result::SUCCESS &&
           data.state == Block::State::Created;
}

bool IsConnected(Flowgraph& flowgraph,
                 const std::string& block,
                 const std::string& input,
                 const std::string& source,
                 const std::string& output) {
    Flowgraph::View::BlockData data;
    if (flowgraph.view().block(block, data) != Result::SUCCESS ||
        !data.inputs.contains(input) ||
        !data.inputs.at(input).external.has_value()) {
        return false;
    }

    const auto& endpoint = data.inputs.at(input).external.value();
    return endpoint.block == source && endpoint.port == output;
}

}  // namespace

using namespace Jetstream;
using namespace TestFlowgraph;

TEST_CASE_METHOD(FlowgraphFixture,
                  "Flowgraph block names are validated before mutation",
                  "[core][flowgraph][mutation][name]") {
    REQUIRE(flowgraph->blockCreate("", kSyntheticSourceType, {}, {}) == Result::ERROR);
    REQUIRE(flowgraph->blockCreate("invalid.name", kSyntheticSourceType, {}, {}) == Result::ERROR);
    REQUIRE(flowgraph->view().empty());
    REQUIRE(GraphKeys(*flowgraph).empty());

    REQUIRE(flowgraph->blockCreate("source-name_1", kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    const auto before = GraphKeys(*flowgraph);

    REQUIRE(flowgraph->blockRename("source-name_1", "") == Result::ERROR);
    REQUIRE(flowgraph->blockRename("source-name_1", "invalid.name") == Result::ERROR);
    REQUIRE(GraphKeys(*flowgraph) == before);
    REQUIRE(flowgraph->view().has("source-name_1"));
    RequireState(*flowgraph, "source-name_1", Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Flowgraph rename rebuilds source middle and leaf topology",
                  "[core][flowgraph][mutation][rename]") {
    CreateChain(*flowgraph);
    SetTag(*flowgraph, "source", "source-tag");
    SetTag(*flowgraph, "middle", "middle-tag");
    SetTag(*flowgraph, "leaf", "leaf-tag");

    SECTION("rename source") {
        REQUIRE(flowgraph->blockRename("source", "renamed-source") == Result::SUCCESS);

        const std::vector<std::string> expected = {"renamed-source", "middle", "leaf"};
        REQUIRE(GraphKeys(*flowgraph) == expected);
        REQUIRE_FALSE(flowgraph->view().has("source"));
        REQUIRE(ViewBlock(*flowgraph, "renamed-source").name == "renamed-source");
        RequireState(*flowgraph, "renamed-source", Block::State::Created);
        RequireState(*flowgraph, "middle", Block::State::Created);
        RequireState(*flowgraph, "leaf", Block::State::Created);
        RequireConnection(*flowgraph, "middle", "buffer", "renamed-source", "signal");
        RequireConnection(*flowgraph, "leaf", "buffer", "middle", "buffer");
        REQUIRE_FALSE(flowgraph->metadata().has("tag", "source"));
        RequireTag(*flowgraph, "renamed-source", "source-tag");
        RequireTag(*flowgraph, "middle", "middle-tag");
        RequireTag(*flowgraph, "leaf", "leaf-tag");
    }

    SECTION("rename middle") {
        REQUIRE(flowgraph->blockRename("middle", "renamed-middle") == Result::SUCCESS);

        const std::vector<std::string> expected = {"source", "renamed-middle", "leaf"};
        REQUIRE(GraphKeys(*flowgraph) == expected);
        REQUIRE_FALSE(flowgraph->view().has("middle"));
        REQUIRE(ViewBlock(*flowgraph, "renamed-middle").name == "renamed-middle");
        RequireState(*flowgraph, "source", Block::State::Created);
        RequireState(*flowgraph, "renamed-middle", Block::State::Created);
        RequireState(*flowgraph, "leaf", Block::State::Created);
        RequireConnection(*flowgraph, "renamed-middle", "buffer", "source", "signal");
        RequireConnection(*flowgraph, "leaf", "buffer", "renamed-middle", "buffer");
        REQUIRE_FALSE(flowgraph->metadata().has("tag", "middle"));
        RequireTag(*flowgraph, "source", "source-tag");
        RequireTag(*flowgraph, "renamed-middle", "middle-tag");
        RequireTag(*flowgraph, "leaf", "leaf-tag");
    }

    SECTION("rename leaf") {
        REQUIRE(flowgraph->blockRename("leaf", "renamed-leaf") == Result::SUCCESS);

        const std::vector<std::string> expected = {"source", "middle", "renamed-leaf"};
        REQUIRE(GraphKeys(*flowgraph) == expected);
        REQUIRE_FALSE(flowgraph->view().has("leaf"));
        REQUIRE(ViewBlock(*flowgraph, "renamed-leaf").name == "renamed-leaf");
        RequireState(*flowgraph, "source", Block::State::Created);
        RequireState(*flowgraph, "middle", Block::State::Created);
        RequireState(*flowgraph, "renamed-leaf", Block::State::Created);
        RequireConnection(*flowgraph, "middle", "buffer", "source", "signal");
        RequireConnection(*flowgraph, "renamed-leaf", "buffer", "middle", "buffer");
        REQUIRE_FALSE(flowgraph->metadata().has("tag", "leaf"));
        RequireTag(*flowgraph, "source", "source-tag");
        RequireTag(*flowgraph, "middle", "middle-tag");
        RequireTag(*flowgraph, "renamed-leaf", "leaf-tag");
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Flowgraph rename return contracts are atomic",
                  "[core][flowgraph][mutation][rename][atomicity]") {
    CreateChain(*flowgraph);
    SetTag(*flowgraph, "middle", "middle-tag");

    SECTION("duplicate target is rejected") {
        REQUIRE(flowgraph->blockRename("middle", "leaf") == Result::ERROR);
        RequireIntactChain(*flowgraph);
        RequireTag(*flowgraph, "middle", "middle-tag");
    }

    SECTION("missing source is rejected") {
        REQUIRE(flowgraph->blockRename("missing", "replacement") == Result::ERROR);
        RequireIntactChain(*flowgraph);
        RequireTag(*flowgraph, "middle", "middle-tag");
    }

    SECTION("same name is a successful no-op") {
        REQUIRE(flowgraph->blockRename("middle", "middle") == Result::SUCCESS);
        RequireIntactChain(*flowgraph);
        RequireTag(*flowgraph, "middle", "middle-tag");
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Flowgraph connect disconnect and destroy propagate topology",
                  "[core][flowgraph][mutation][propagation]") {
    CreateChain(*flowgraph);
    REQUIRE(flowgraph->blockCreate("replacement", kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    SetTag(*flowgraph, "middle", "middle-tag");
    SetTag(*flowgraph, "leaf", "leaf-tag");

    SECTION("connect rebuilds the target and downstream chain") {
        REQUIRE(flowgraph->blockConnect("middle", "buffer", "replacement", "signal") ==
                Result::SUCCESS);

        RequireState(*flowgraph, "source", Block::State::Created);
        RequireState(*flowgraph, "replacement", Block::State::Created);
        RequireState(*flowgraph, "middle", Block::State::Created);
        RequireState(*flowgraph, "leaf", Block::State::Created);
        RequireConnection(*flowgraph, "middle", "buffer", "replacement", "signal");
        RequireConnection(*flowgraph, "leaf", "buffer", "middle", "buffer");
        RequireTag(*flowgraph, "middle", "middle-tag");
        RequireTag(*flowgraph, "leaf", "leaf-tag");
    }

    SECTION("disconnect invalidates downstream and reconnect restores it") {
        REQUIRE(flowgraph->blockDisconnect("middle", "buffer") == Result::SUCCESS);

        RequireState(*flowgraph, "middle", Block::State::Incomplete);
        RequireState(*flowgraph, "leaf", Block::State::Incomplete);
        RequireNoConnection(*flowgraph, "middle", "buffer");
        RequireConnection(*flowgraph, "leaf", "buffer", "middle", "buffer");
        RequireTag(*flowgraph, "middle", "middle-tag");
        RequireTag(*flowgraph, "leaf", "leaf-tag");

        REQUIRE(flowgraph->blockConnect("middle", "buffer", "source", "signal") ==
                Result::SUCCESS);
        RequireState(*flowgraph, "middle", Block::State::Created);
        RequireState(*flowgraph, "leaf", Block::State::Created);
        RequireConnection(*flowgraph, "middle", "buffer", "source", "signal");
        RequireConnection(*flowgraph, "leaf", "buffer", "middle", "buffer");
    }

    SECTION("destroy source severs direct links and invalidates descendants") {
        REQUIRE(flowgraph->blockDestroy("source") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->view().has("source"));
        RequireState(*flowgraph, "replacement", Block::State::Created);
        RequireState(*flowgraph, "middle", Block::State::Incomplete);
        RequireState(*flowgraph, "leaf", Block::State::Incomplete);
        RequireNoConnection(*flowgraph, "middle", "buffer");
        RequireConnection(*flowgraph, "leaf", "buffer", "middle", "buffer");
    }

    SECTION("destroy middle severs the leaf link without disturbing sources") {
        REQUIRE(flowgraph->blockDestroy("middle") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->view().has("middle"));
        RequireState(*flowgraph, "source", Block::State::Created);
        RequireState(*flowgraph, "replacement", Block::State::Created);
        RequireState(*flowgraph, "leaf", Block::State::Incomplete);
        RequireNoConnection(*flowgraph, "leaf", "buffer");
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                   "Rejected flowgraph mutations leave topology unchanged",
                   "[core][flowgraph][mutation][atomicity]") {
    CreateChain(*flowgraph);

    SECTION("connect rejects missing target") {
        REQUIRE(flowgraph->blockConnect("missing", "buffer", "source", "signal") == Result::ERROR);
        RequireIntactChain(*flowgraph);
    }

    SECTION("connect rejects missing source") {
        REQUIRE(flowgraph->blockConnect("middle", "buffer", "missing", "signal") == Result::ERROR);
        RequireIntactChain(*flowgraph);
    }

    SECTION("disconnect rejects missing input") {
        REQUIRE(flowgraph->blockDisconnect("middle", "missing") == Result::ERROR);
        RequireIntactChain(*flowgraph);
    }

    SECTION("destroy rejects missing block") {
        REQUIRE(flowgraph->blockDestroy("missing") == Result::ERROR);
        RequireIntactChain(*flowgraph);
    }

    SECTION("create rejects a missing input source without reserving the name") {
        REQUIRE(flowgraph->blockCreate("orphan", kSyntheticPassType, {},
                                      RequestedInput("buffer", "missing", "buffer")) == Result::ERROR);
        RequireIntactChain(*flowgraph);
        REQUIRE_FALSE(flowgraph->view().has("orphan"));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Faulted flowgraph mutations roll back block state",
                 "[core][flowgraph][mutation][fault][atomicity]") {
    auto& faults = syntheticFaultState();
    REQUIRE(flowgraph->blockCreate("fault", kSyntheticFaultType, {}, {}) == Result::SUCCESS);
    const auto original = viewBlock("fault");
    REQUIRE(std::any_cast<U64>(original.config.at("revision")) == 0);

    SECTION("module destroy failure retains a usable block") {
        faults.failNext(SyntheticFaultPoint::ModuleDestroy);
        REQUIRE(flowgraph->blockDestroy("fault") == Result::ERROR);

        const auto retained = viewBlock("fault");
        REQUIRE(retained.type == kSyntheticFaultType);
        REQUIRE(std::any_cast<U64>(retained.config.at("revision")) == 0);
        REQUIRE(retained.outputs.contains("out"));
        REQUIRE(retained.outputs.at("out").resolved());
        // Defect: failed module destruction restores an errored block with its module removed.
        CHECK(retained.state == Block::State::Created);
        REQUIRE(faults.moduleDestroyCalls == 1);

        REQUIRE(flowgraph->blockDestroy("fault") == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->view().has("fault"));
    }

    SECTION("block configure failure restores the committed configuration") {
        Parser::Map update;
        update["revision"] = U64{1};
        faults.failNext(SyntheticFaultPoint::BlockConfigure);
        REQUIRE(flowgraph->blockReconfigure("fault", update) == Result::ERROR);

        const auto retained = viewBlock("fault");
        REQUIRE(retained.state == Block::State::Created);
        // Defect: failed block configuration leaves the staged block config mutated.
        CHECK(std::any_cast<U64>(retained.config.at("revision")) == 0);
        REQUIRE(faults.blockConfigureCalls == 2);
    }

    SECTION("module reconfigure failure restores the committed configuration") {
        Parser::Map update;
        update["revision"] = U64{1};
        faults.failNext(SyntheticFaultPoint::ModuleReconfigure);
        REQUIRE(flowgraph->blockReconfigure("fault", update) == Result::ERROR);

        const auto retained = viewBlock("fault");
        REQUIRE(retained.state == Block::State::Created);
        // Defect: failed module reconfiguration leaves the staged block config mutated.
        CHECK(std::any_cast<U64>(retained.config.at("revision")) == 0);
        REQUIRE(faults.moduleReconfigureCalls == 1);
    }

    SECTION("cascading recreation failure restores destroyed descendants") {
        REQUIRE(flowgraph->blockCreate("leaf", kSyntheticPassType, {},
                                       RequestedInput("buffer", "fault", "out")) == Result::SUCCESS);
        const auto before = GraphKeys(*flowgraph);

        faults.failNext(SyntheticFaultPoint::ModuleDestroy);
        REQUIRE(flowgraph->blockRecreate("fault", original.config) == Result::ERROR);

        // Defect: a failed target destroy does not restore descendants removed earlier in the mutation.
        CHECK((GraphKeys(*flowgraph) == before &&
               IsCreated(*flowgraph, "fault") &&
               IsCreated(*flowgraph, "leaf") &&
               IsConnected(*flowgraph, "leaf", "buffer", "fault", "out")));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Replacing a connection removes the old dependency edge",
                 "[core][flowgraph][mutation][edge]") {
    REQUIRE(flowgraph->blockCreate("old-source", kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("replacement", kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("middle", kSyntheticPassType, {},
                                   RequestedInput("buffer", "old-source", "signal")) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("leaf", kSyntheticPassType, {},
                                   RequestedInput("buffer", "middle", "buffer")) == Result::SUCCESS);

    REQUIRE(flowgraph->blockConnect("middle", "buffer", "replacement", "signal") == Result::SUCCESS);
    const auto middleOutputId = viewBlock("middle").outputs.at("buffer").tensor.id();
    const auto leafOutputId = viewBlock("leaf").outputs.at("buffer").tensor.id();

    REQUIRE(flowgraph->blockDestroy("old-source") == Result::SUCCESS);

    const std::vector<std::string> expected = {"replacement", "middle", "leaf"};
    REQUIRE(GraphKeys(*flowgraph) == expected);
    RequireState(*flowgraph, "middle", Block::State::Created);
    RequireState(*flowgraph, "leaf", Block::State::Created);
    RequireConnection(*flowgraph, "middle", "buffer", "replacement", "signal");
    RequireConnection(*flowgraph, "leaf", "buffer", "middle", "buffer");
    REQUIRE(viewBlock("middle").outputs.at("buffer").tensor.id() == middleOutputId);
    REQUIRE(viewBlock("leaf").outputs.at("buffer").tensor.id() == leafOutputId);
}

TEST_CASE_METHOD(FlowgraphFixture,
                   "Flowgraph recreation uses dependency order",
                  "[core][flowgraph][mutation][recreation]") {
    REQUIRE(flowgraph->blockCreate("source", kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("short", kSyntheticPassType, {},
                                  RequestedInput("buffer", "source", "signal")) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("long-1", kSyntheticPassType, {},
                                  RequestedInput("buffer", "source", "signal")) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("long-2", kSyntheticPassType, {},
                                  RequestedInput("buffer", "long-1", "buffer")) == Result::SUCCESS);

    TensorMap mergeInputs;
    mergeInputs["a"].requested("short", "buffer");
    mergeInputs["b"].requested("long-2", "buffer");
    REQUIRE(flowgraph->blockCreate("merge", kSyntheticMergeType, {}, mergeInputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("spectator", kSyntheticSourceType, {}, {}) == Result::SUCCESS);

    const auto sourceConfig = ViewBlock(*flowgraph, "source").config;
    REQUIRE(flowgraph->blockRecreate("source", sourceConfig) == Result::SUCCESS);

    const std::vector<std::string> expected = {
        "spectator", "source", "short", "long-1", "long-2", "merge",
    };
    REQUIRE(GraphKeys(*flowgraph) == expected);
    for (const auto* name : {"spectator", "source", "short", "long-1", "long-2", "merge"}) {
        RequireState(*flowgraph, name, Block::State::Created);
    }
    RequireConnection(*flowgraph, "short", "buffer", "source", "signal");
    RequireConnection(*flowgraph, "long-1", "buffer", "source", "signal");
    RequireConnection(*flowgraph, "long-2", "buffer", "long-1", "buffer");
    RequireConnection(*flowgraph, "merge", "a", "short", "buffer");
    RequireConnection(*flowgraph, "merge", "b", "long-2", "buffer");
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Cycle-producing connections are rejected atomically",
                  "[core][flowgraph][mutation][cycle][atomicity]") {
    CreateChain(*flowgraph);
    const auto before = GraphKeys(*flowgraph);
    const auto sourceOutputId = ViewBlock(*flowgraph, "source").outputs.at("signal").tensor.id();
    const auto middleOutputId = ViewBlock(*flowgraph, "middle").outputs.at("buffer").tensor.id();
    const auto leafOutputId = ViewBlock(*flowgraph, "leaf").outputs.at("buffer").tensor.id();

    SECTION("self-cycle") {
        REQUIRE(flowgraph->blockConnect("middle", "buffer", "middle", "buffer") == Result::ERROR);
        const auto after = GraphKeys(*flowgraph);

        REQUIRE(after == before);
        RequireIntactChain(*flowgraph);
        REQUIRE(ViewBlock(*flowgraph, "source").outputs.at("signal").tensor.id() == sourceOutputId);
        REQUIRE(ViewBlock(*flowgraph, "middle").outputs.at("buffer").tensor.id() == middleOutputId);
        REQUIRE(ViewBlock(*flowgraph, "leaf").outputs.at("buffer").tensor.id() == leafOutputId);
    }

    SECTION("back edge") {
        REQUIRE(flowgraph->blockConnect("middle", "buffer", "leaf", "buffer") == Result::ERROR);
        const auto after = GraphKeys(*flowgraph);

        REQUIRE(after == before);
        RequireIntactChain(*flowgraph);
        REQUIRE(ViewBlock(*flowgraph, "source").outputs.at("signal").tensor.id() == sourceOutputId);
        REQUIRE(ViewBlock(*flowgraph, "middle").outputs.at("buffer").tensor.id() == middleOutputId);
        REQUIRE(ViewBlock(*flowgraph, "leaf").outputs.at("buffer").tensor.id() == leafOutputId);
    }
}
