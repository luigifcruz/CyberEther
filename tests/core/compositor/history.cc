#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/base.hh"
#include "flowgraph_fixture.hh"

#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace Jetstream;

namespace {

struct HistoryFixture : FlowgraphFixture {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    HistoryActions history{state, callbacks};
    FlowgraphActions flowgraphActions{state, callbacks};
    std::vector<Mail> mail;
    std::vector<Result> notified;
    std::deque<std::function<void()>> commands;
    FlowgraphEdit::Clock::time_point now = FlowgraphEdit::Clock::now();

    HistoryFixture() {
        state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
        callbacks.enqueueMail = [this](Mail&& message) {
            mail.push_back(std::move(message));
        };
        callbacks.notifyResult = [this](Result result, const std::string&) {
            notified.push_back(result);
        };
        callbacks.enqueueCommandWithCompletion = [this](Sakura::Callback<Result()> command,
                                                        bool,
                                                        Sakura::Callback<void(Result)> onComplete) {
            commands.push_back([command = std::move(command), onComplete = std::move(onComplete)]() {
                const Result result = command();
                if (onComplete) {
                    onComplete(result);
                }
            });
        };
    }

    ~HistoryFixture() {
        state.flowgraph.history.clear();
    }

    void start() {
        history.reconcile();
        drain();
    }

    void drain() {
        while (!commands.empty()) {
            auto next = std::move(commands.front());
            commands.pop_front();
            next();
        }
    }

    void request(std::function<Result()> command, std::string mergeKey = "") {
        history.enqueueEdit({
            .flowgraph = "graph",
            .mergeKey = std::move(mergeKey),
            .command = std::move(command),
            .requestedAt = now,
        });
    }

    void edit(std::function<Result()> command, std::string mergeKey = "") {
        request(std::move(command), std::move(mergeKey));
        drain();
    }

    void arrange() {
        REQUIRE(history.handle(MailRecordFlowgraphLayout{.flowgraph = "graph"}) == Result::SUCCESS);
        drain();
    }

    void undo() {
        REQUIRE(history.handle(MailUndoFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
        drain();
    }

    void redo() {
        REQUIRE(history.handle(MailRedoFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
        drain();
    }

    FlowgraphHistory& graphHistory() {
        return state.flowgraph.history.at("graph");
    }

    std::vector<std::string> blocks() {
        std::vector<std::string> names;
        REQUIRE(flowgraph->view().keys(names) == Result::SUCCESS);
        std::sort(names.begin(), names.end());
        return names;
    }

    TensorMap inputFrom(const std::string& source, const std::string& port) {
        TensorMap inputs;
        inputs["buffer"].requested(source, port);
        return inputs;
    }

    Result create(const std::string& name, const char* type = TestFlowgraph::kSyntheticIsolatedType) {
        return flowgraph->blockCreate(name, type, {}, {});
    }

    void createChain() {
        REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("middle", TestFlowgraph::kSyntheticPassType, {},
                                       inputFrom("source", "signal")) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("leaf", TestFlowgraph::kSyntheticPassType, {},
                                       inputFrom("middle", "buffer")) == Result::SUCCESS);
    }

    std::string sourceOf(const std::string& block) {
        const auto data = viewBlock(block);
        const auto input = data.inputs.find("buffer");
        if (input == data.inputs.end() || !input->second.external.has_value()) {
            return "";
        }
        return input->second.external->block;
    }

    Result setNode(const std::string& block, NodeMeta meta) {
        return flowgraph->metadata().set("node", meta, block);
    }

    void placeNode(const std::string& block, NodeMeta meta) {
        REQUIRE(flowgraphActions.handle(MailSetNodeMeta{.flowgraph = "graph", .block = block, .meta = meta}) ==
                Result::SUCCESS);
    }

    NodeMeta node(const std::string& block) {
        NodeMeta meta;
        REQUIRE(flowgraph->metadata().get("node", meta, block) == Result::SUCCESS);
        return meta;
    }

    Result setSourceValue(F32 value) {
        Parser::Map config;
        config["value"] = value;
        return flowgraph->blockReconfigure("source", config);
    }

    F32 sourceValue() {
        F32 value = 0.0f;
        REQUIRE(Parser::Deserialize(viewBlock("source").config, "value", value) == Result::SUCCESS);
        return value;
    }
};

}  // namespace

TEST_CASE_METHOD(HistoryFixture, "A flowgraph starts its history from the state it opens with",
                 "[core][compositor][history]") {
    createChain();
    start();
    CHECK(graphHistory().entries.size() == 1);
    CHECK_FALSE(graphHistory().canUndo());
    CHECK_FALSE(graphHistory().canRedo());

    undo();
    redo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
}

TEST_CASE_METHOD(HistoryFixture, "Undo and redo walk through created blocks and restore their placement",
                 "[core][compositor][history]") {
    start();
    edit([&] {
        JST_CHECK(setNode("source", NodeMeta{120.0f, 80.0f, 0.0f, 0.0f}));
        return flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {});
    });
    REQUIRE(graphHistory().canUndo());

    undo();
    CHECK(blocks().empty());
    CHECK(graphHistory().canRedo());

    redo();
    CHECK(blocks() == std::vector<std::string>{"source"});
    CHECK(node("source").x == 120.0f);
    CHECK(node("source").y == 80.0f);
    CHECK_FALSE(graphHistory().canRedo());
}

TEST_CASE_METHOD(HistoryFixture, "Undoing a delete brings back the block, its config and its links both ways",
                 "[core][compositor][history]") {
    createChain();
    REQUIRE(setSourceValue(7.0f) == Result::SUCCESS);
    start();

    edit([&] {
        JST_CHECK(flowgraph->blockDestroy("source"));
        return flowgraph->blockDestroy("middle");
    });
    REQUIRE(blocks() == std::vector<std::string>{"leaf"});
    REQUIRE(sourceOf("leaf").empty());

    undo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
    CHECK(sourceOf("middle") == "source");
    CHECK(sourceOf("leaf") == "middle");
    CHECK(sourceValue() == 7.0f);
    CHECK(viewBlock("leaf").state == Block::State::Created);

    redo();
    CHECK(blocks() == std::vector<std::string>{"leaf"});
    CHECK(sourceOf("leaf").empty());
}

TEST_CASE_METHOD(HistoryFixture, "Undo reverts link edits without an impossible intermediate cycle",
                 "[core][compositor][history]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("first", TestFlowgraph::kSyntheticPassType, {},
                                   inputFrom("source", "signal")) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("second", TestFlowgraph::kSyntheticPassType, {},
                                   inputFrom("first", "buffer")) == Result::SUCCESS);
    start();

    edit([&] {
        JST_CHECK(flowgraph->blockDisconnect("second", "buffer"));
        JST_CHECK(flowgraph->blockDisconnect("first", "buffer"));
        JST_CHECK(flowgraph->blockConnect("second", "buffer", "source", "signal"));
        return flowgraph->blockConnect("first", "buffer", "second", "buffer");
    });

    undo();
    CHECK(notified.empty());
    CHECK(sourceOf("first") == "source");
    CHECK(sourceOf("second") == "first");

    redo();
    CHECK(notified.empty());
    CHECK(sourceOf("second") == "source");
    CHECK(sourceOf("first") == "second");
}

TEST_CASE_METHOD(HistoryFixture, "Undo and redo restore the config of a block and keep its links",
                 "[core][compositor][history]") {
    createChain();
    const F32 original = sourceValue();
    start();

    edit([&] { return setSourceValue(9.0f); });
    REQUIRE(sourceValue() == 9.0f);

    undo();
    CHECK(sourceValue() == original);
    CHECK(sourceOf("middle") == "source");

    redo();
    CHECK(sourceValue() == 9.0f);
    CHECK(sourceOf("leaf") == "middle");
}

TEST_CASE_METHOD(HistoryFixture, "Undoing a rename renames back and moves the tiles of the block",
                 "[core][compositor][history]") {
    createChain();
    start();

    edit([&] { return flowgraph->blockRename("middle", "renamed"); });
    REQUIRE(sourceOf("leaf") == "renamed");

    undo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
    CHECK(sourceOf("leaf") == "middle");
    REQUIRE(mail.size() == 1);
    const auto* rename = std::get_if<MailRenameLayoutBlock>(&mail.front());
    REQUIRE(rename != nullptr);
    CHECK(rename->oldId == "renamed");
    CHECK(rename->newId == "middle");
}

TEST_CASE_METHOD(HistoryFixture, "Undoing a new block removes its tiles", "[core][compositor][history]") {
    start();
    edit([&] { return create("source", TestFlowgraph::kSyntheticSourceType); });

    undo();
    REQUIRE(mail.size() == 1);
    const auto* removed = std::get_if<MailRemoveLayoutBlock>(&mail.front());
    REQUIRE(removed != nullptr);
    CHECK(removed->block == "source");
}

TEST_CASE_METHOD(HistoryFixture, "Undo brings back a block replaced by another type under the same name",
                 "[core][compositor][history]") {
    createChain();
    start();

    edit([&] {
        JST_CHECK(flowgraph->blockDestroy("middle"));
        return create("middle");
    });
    REQUIRE(sourceOf("leaf").empty());

    undo();
    CHECK(notified.empty());
    CHECK(viewBlock("middle").type == TestFlowgraph::kSyntheticPassType);
    CHECK(sourceOf("middle") == "source");
    CHECK(sourceOf("leaf") == "middle");

    redo();
    CHECK(notified.empty());
    CHECK(viewBlock("middle").type == TestFlowgraph::kSyntheticIsolatedType);
    CHECK(sourceOf("leaf").empty());
}

TEST_CASE_METHOD(HistoryFixture, "Arranging nodes is an undo step that doesn't rebuild the blocks",
                 "[core][compositor][history]") {
    createChain();
    REQUIRE(setNode("leaf", NodeMeta{10.0f, 20.0f, 200.0f, 100.0f}) == Result::SUCCESS);
    start();
    const auto tensor = viewBlock("middle").outputs.at("buffer").tensor.id();

    REQUIRE(setNode("leaf", NodeMeta{300.0f, 400.0f, 320.0f, 100.0f}) == Result::SUCCESS);
    CHECK_FALSE(graphHistory().canUndo());
    arrange();
    REQUIRE(graphHistory().canUndo());

    undo();
    CHECK(node("leaf").x == 10.0f);
    CHECK(node("leaf").width == 200.0f);
    CHECK(viewBlock("middle").outputs.at("buffer").tensor.id() == tensor);

    redo();
    CHECK(node("leaf").x == 300.0f);
    CHECK(node("leaf").width == 320.0f);

    arrange();
    CHECK(graphHistory().entries.size() == 2);
}

TEST_CASE_METHOD(HistoryFixture, "Placement and sizes the editor sets on its own are not undo steps",
                 "[core][compositor][history]") {
    createChain();
    REQUIRE(setNode("leaf", NodeMeta{10.0f, 20.0f, 0.0f, 0.0f}) == Result::SUCCESS);
    const F32 original = sourceValue();
    start();

    REQUIRE(setNode("leaf", NodeMeta{10.0f, 20.0f, 220.0f, 140.0f}) == Result::SUCCESS);
    REQUIRE(setNode("source", NodeMeta{500.0f, 60.0f, 0.0f, 0.0f}) == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("config", ConfigMeta{.collapsed = true}, "leaf") == Result::SUCCESS);
    edit([&] { return setSourceValue(9.0f); });
    CHECK(graphHistory().entries.size() == 2);

    undo();
    CHECK(sourceValue() == original);
    CHECK(node("leaf").width == 220.0f);
    CHECK_FALSE(graphHistory().canUndo());
}

TEST_CASE_METHOD(HistoryFixture, "Quick edits of the same fields merge into one undo step",
                 "[core][compositor][history]") {
    createChain();
    const F32 original = sourceValue();
    start();

    for (const F32 value : {1.0f, 2.0f, 3.0f}) {
        now += HistoryActions::MergeWindow / 4;
        edit([&, value] { return setSourceValue(value); }, "source/value");
    }
    CHECK(graphHistory().entries.size() == 2);

    now += HistoryActions::MergeWindow * 2;
    edit([&] { return setSourceValue(4.0f); }, "source/value");
    CHECK(graphHistory().entries.size() == 3);

    edit([&] { return setSourceValue(5.0f); }, "source/other");
    CHECK(graphHistory().entries.size() == 4);

    undo();
    undo();
    CHECK(sourceValue() == 3.0f);
    undo();
    CHECK(sourceValue() == original);
}

TEST_CASE_METHOD(HistoryFixture, "A new edit after an undo discards what could be redone",
                 "[core][compositor][history]") {
    start();
    edit([&] { return create("a"); });
    edit([&] { return create("b"); });

    undo();
    REQUIRE(graphHistory().canRedo());
    REQUIRE(blocks() == std::vector<std::string>{"a"});

    edit([&] { return create("c"); });
    CHECK_FALSE(graphHistory().canRedo());

    undo();
    CHECK(blocks() == std::vector<std::string>{"a"});
    undo();
    CHECK(blocks().empty());
}

TEST_CASE_METHOD(HistoryFixture, "Undo includes the edits that were requested before it",
                 "[core][compositor][history]") {
    start();
    edit([&] { return create("a"); });

    request([&] { return create("b"); });
    REQUIRE(history.handle(MailUndoFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
    drain();
    CHECK(blocks() == std::vector<std::string>{"a"});
    CHECK(graphHistory().canRedo());
}

TEST_CASE_METHOD(HistoryFixture, "Presses during a restore apply together and edits wait for it",
                 "[core][compositor][history]") {
    start();
    for (const std::string name : {"a", "b", "c"}) {
        edit([&, name] { return create(name); });
    }

    for (U64 i = 0; i < 3; ++i) {
        REQUIRE(history.handle(MailUndoFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
    }
    request([&] { return create("d"); });
    CHECK(graphHistory().deferred.size() == 1);

    drain();
    CHECK(blocks() == std::vector<std::string>{"d"});
    CHECK_FALSE(graphHistory().canRedo());

    undo();
    CHECK(blocks().empty());
}

TEST_CASE_METHOD(HistoryFixture, "Undo and edits requested during a restore apply in the order they were made",
                 "[core][compositor][history]") {
    start();
    edit([&] { return create("a"); });
    edit([&] { return create("b"); });

    REQUIRE(history.handle(MailUndoFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
    request([&] { return create("c"); });
    REQUIRE(history.handle(MailUndoFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
    drain();
    CHECK(blocks() == std::vector<std::string>{"a"});
    REQUIRE(graphHistory().canRedo());

    redo();
    CHECK(blocks() == std::vector<std::string>{"a", "c"});
}

TEST_CASE_METHOD(HistoryFixture, "The first drag of a node placed by the editor is an undo step",
                 "[core][compositor][history]") {
    REQUIRE(create("a") == Result::SUCCESS);
    start();
    placeNode("a", NodeMeta{40.0f, 60.0f, 200.0f, 100.0f});
    CHECK_FALSE(graphHistory().canUndo());

    placeNode("a", NodeMeta{300.0f, 400.0f, 200.0f, 100.0f});
    arrange();
    REQUIRE(graphHistory().canUndo());

    undo();
    CHECK(node("a").x == 40.0f);
    CHECK(node("a").y == 60.0f);

    redo();
    CHECK(node("a").x == 300.0f);
}

TEST_CASE_METHOD(HistoryFixture, "Undo brings a deleted block back where the editor placed it",
                 "[core][compositor][history]") {
    REQUIRE(create("a") == Result::SUCCESS);
    start();
    placeNode("a", NodeMeta{40.0f, 60.0f, 200.0f, 100.0f});
    edit([&] { return flowgraph->blockDestroy("a"); });

    undo();
    REQUIRE(blocks() == std::vector<std::string>{"a"});
    CHECK(node("a").x == 40.0f);
    CHECK(node("a").y == 60.0f);
}

TEST_CASE_METHOD(HistoryFixture, "Deleting several blocks at once is one undo step and drops each of their tiles",
                 "[core][compositor][history]") {
    createChain();
    start();
    callbacks.enqueueFlowgraphEdit = [this](FlowgraphEdit edit) {
        history.enqueueEdit(std::move(edit));
    };

    REQUIRE(flowgraphActions.handle(MailDeleteBlock{.flowgraph = "graph",
                                                    .blockIds = {"source", "middle", "source", ""}}) ==
            Result::SUCCESS);
    drain();
    REQUIRE(blocks() == std::vector<std::string>{"leaf"});

    std::vector<std::string> removed;
    for (const auto& message : mail) {
        if (const auto* remove = std::get_if<MailRemoveLayoutBlock>(&message)) {
            removed.push_back(remove->block);
        }
    }
    CHECK(removed == std::vector<std::string>{"source", "middle"});

    undo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
    CHECK(sourceOf("middle") == "source");
    CHECK(sourceOf("leaf") == "middle");
    CHECK_FALSE(graphHistory().canUndo());

    redo();
    CHECK(blocks() == std::vector<std::string>{"leaf"});
}

TEST_CASE_METHOD(HistoryFixture, "A failed multi-delete keeps the blocks it removed in one undo step",
                 "[core][compositor][history]") {
    createChain();
    start();
    callbacks.enqueueFlowgraphEdit = [this](FlowgraphEdit edit) {
        history.enqueueEdit(std::move(edit));
    };

    REQUIRE(flowgraphActions.handle(MailDeleteBlock{.flowgraph = "graph", .blockIds = {"leaf", "missing", "source"}}) ==
            Result::SUCCESS);
    drain();
    REQUIRE(blocks() == std::vector<std::string>{"middle", "source"});

    std::vector<std::string> removed;
    for (const auto& message : mail) {
        if (const auto* remove = std::get_if<MailRemoveLayoutBlock>(&message)) {
            removed.push_back(remove->block);
        }
    }
    CHECK(removed == std::vector<std::string>{"leaf"});

    undo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
    CHECK_FALSE(graphHistory().canUndo());
}

TEST_CASE_METHOD(HistoryFixture, "An edit made outside the compositor becomes its own undo step",
                 "[core][compositor][history]") {
    start();
    REQUIRE(create("outside") == Result::SUCCESS);
    edit([&] { return create("inside"); });
    CHECK(graphHistory().entries.size() == 3);

    undo();
    CHECK(blocks() == std::vector<std::string>{"outside"});
    undo();
    CHECK(blocks().empty());
}

TEST_CASE_METHOD(HistoryFixture, "Undo right after an edit made outside the compositor reverts only that edit",
                 "[core][compositor][history]") {
    start();
    edit([&] { return create("inside"); });
    REQUIRE(create("outside") == Result::SUCCESS);

    undo();
    CHECK(blocks() == std::vector<std::string>{"inside"});
    REQUIRE(graphHistory().canRedo());

    redo();
    CHECK(blocks() == std::vector<std::string>{"inside", "outside"});

    undo();
    undo();
    CHECK(blocks().empty());
}

TEST_CASE_METHOD(HistoryFixture, "Redo after an edit made outside the compositor keeps that edit",
                 "[core][compositor][history]") {
    start();
    edit([&] { return create("a"); });
    undo();
    REQUIRE(graphHistory().canRedo());
    REQUIRE(create("outside") == Result::SUCCESS);

    redo();
    CHECK(blocks() == std::vector<std::string>{"outside"});
    CHECK_FALSE(graphHistory().canRedo());

    undo();
    CHECK(blocks().empty());
}

TEST_CASE_METHOD(HistoryFixture, "A restore reports blocks it cannot order instead of dropping them",
                 "[core][compositor][history]") {
    start();
    edit([&] { return create("a"); });

    auto looped = std::make_shared<FlowgraphSnapshot>(*graphHistory().entries[0]);
    FlowgraphSnapshot::Block self;
    self.name = "self";
    self.type = TestFlowgraph::kSyntheticPassType;
    self.inputs["buffer"] = {.block = "self", .port = "buffer"};
    looped->blocks.push_back(self);
    graphHistory().entries[0] = looped;

    undo();
    CHECK(blocks().empty());
    REQUIRE(notified.size() == 1);
    CHECK(notified.front() != Result::SUCCESS);
}

TEST_CASE_METHOD(HistoryFixture, "A failed restore reports the error and keeps the history usable",
                 "[core][compositor][history]") {
    start();
    edit([&] { return create("a", TestFlowgraph::kSyntheticSourceType); });

    auto broken = std::make_shared<FlowgraphSnapshot>(*graphHistory().entries[0]);
    FlowgraphSnapshot::Block unknown;
    unknown.name = "unknown";
    unknown.type = "flowgraph_test_type_that_does_not_exist";
    broken->blocks.push_back(unknown);
    graphHistory().entries[0] = broken;

    undo();
    REQUIRE(notified.size() == 1);
    CHECK(notified.front() != Result::SUCCESS);
    CHECK(blocks().empty());
    REQUIRE(graphHistory().canRedo());

    redo();
    CHECK(blocks() == std::vector<std::string>{"a"});
}

TEST_CASE_METHOD(HistoryFixture, "The history stays within its capacity and forgets closed flowgraphs",
                 "[core][compositor][history]") {
    REQUIRE(setNode("a", NodeMeta{}) == Result::SUCCESS);
    REQUIRE(create("a") == Result::SUCCESS);
    start();
    for (U64 i = 1; i <= FlowgraphHistory::Capacity + 8; ++i) {
        REQUIRE(setNode("a", NodeMeta{static_cast<F32>(i) * 10.0f, 0.0f, 0.0f, 0.0f}) == Result::SUCCESS);
        arrange();
    }
    CHECK(graphHistory().entries.size() == FlowgraphHistory::Capacity);
    CHECK(graphHistory().cursor == FlowgraphHistory::Capacity - 1);

    state.flowgraph.items.erase("graph");
    history.reconcile();
    CHECK_FALSE(state.flowgraph.history.contains("graph"));
}

TEST_CASE("Config fingerprints follow values, ignore entry order and stay quiet on opaque values",
          "[core][compositor][history]") {
    const auto fingerprint = [](const Parser::Map& map) {
        return FlowgraphValueFingerprint(std::any(map));
    };

    Parser::Map ordered;
    ordered["a"] = F32{1.0f};
    ordered["b"] = std::string("text");
    Parser::Map reversed;
    reversed["b"] = std::string("text");
    reversed["a"] = F32{1.0f};
    CHECK(fingerprint(ordered) == fingerprint(reversed));

    Parser::Map changed = ordered;
    changed["a"] = F32{2.0f};
    CHECK(fingerprint(ordered) != fingerprint(changed));

    Parser::Map typed = ordered;
    typed["a"] = F64{1.0};
    CHECK(fingerprint(ordered) != fingerprint(typed));

    Parser::Map nested;
    nested["child"] = ordered;
    Parser::Map nestedChanged;
    nestedChanged["child"] = changed;
    CHECK(fingerprint(nested) != fingerprint(nestedChanged));

    JST_ERROR("[TEST] Sentinel.");
    const std::string before = JST_LOG_LAST_ERROR();

    Parser::Map opaque;
    opaque["buffer"] = Tensor{};
    opaque["a"] = F32{1.0f};
    Parser::Map opaqueChanged = opaque;
    opaqueChanged["a"] = F32{2.0f};
    CHECK(fingerprint(opaque) == fingerprint(opaque));
    CHECK(fingerprint(opaque) != fingerprint(opaqueChanged));
    CHECK(JST_LOG_LAST_ERROR() == before);
}

TEST_CASE("Block edits from the default dispatch are undone and redone through it",
          "[core][compositor][history]") {
    REQUIRE(TestFlowgraph::RegisterSyntheticGraph() == Result::SUCCESS);
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    AgentRuntime runtime{state, callbacks};
    DefaultActions actions{state, callbacks, runtime};
    std::deque<std::function<void()>> commands;
    callbacks.enqueueMail = [](Mail&&) {};
    callbacks.enqueueCommandWithCompletion = [&](Sakura::Callback<Result()> command,
                                                 bool,
                                                 Sakura::Callback<void(Result)> onComplete) {
        commands.push_back([command = std::move(command), onComplete = std::move(onComplete)]() {
            const Result result = command();
            if (onComplete) {
                onComplete(result);
            }
        });
    };
    callbacks.enqueueFlowgraphEdit = [&](FlowgraphEdit edit) {
        actions.enqueueFlowgraphEdit(std::move(edit));
    };
    const auto drain = [&]() {
        while (!commands.empty()) {
            auto next = std::move(commands.front());
            commands.pop_front();
            next();
        }
    };
    const auto blockCount = [](Flowgraph& flowgraph) {
        std::vector<std::string> names;
        REQUIRE(flowgraph.view().keys(names) == Result::SUCCESS);
        return names.size();
    };

    auto flowgraph = std::make_shared<Flowgraph>();
    REQUIRE(flowgraph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    state.flowgraph.items["graph"] = flowgraph;
    actions.reconcileHistory();
    drain();

    REQUIRE(actions.handle(MailCreateBlock{
        .flowgraph = "graph",
        .moduleId = TestFlowgraph::kSyntheticSourceType,
        .gridPosition = Extent2D<F32>{40.0f, 60.0f},
        .device = DeviceType::CPU,
        .runtime = RuntimeType::NATIVE,
    }) == Result::SUCCESS);
    drain();
    REQUIRE(blockCount(*flowgraph) == 1);

    REQUIRE(actions.handle(MailUndoFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
    drain();
    CHECK(blockCount(*flowgraph) == 0);

    REQUIRE(actions.handle(MailRedoFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
    drain();
    CHECK(blockCount(*flowgraph) == 1);

    REQUIRE(actions.handle(MailUndoFlowgraph{.flowgraph = "missing"}) == Result::SUCCESS);
    CHECK_FALSE(state.flowgraph.history.contains("missing"));

    state.flowgraph.history.clear();
    std::vector<std::string> names;
    REQUIRE(flowgraph->view().keys(names) == Result::SUCCESS);
    for (const auto& name : names) {
        REQUIRE(flowgraph->blockDestroy(name, false) == Result::SUCCESS);
    }
    REQUIRE(flowgraph->destroy() == Result::SUCCESS);
}
