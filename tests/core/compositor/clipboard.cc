#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/base.hh"
#include "compositor/default/support/clipboard.hh"
#include "flowgraph_fixture.hh"
#include "tools/naming.hh"

#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

using namespace Jetstream;

namespace {

struct ClipboardFixture : FlowgraphFixture {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    HistoryActions history{state, callbacks};
    FlowgraphActions flowgraphActions{state, callbacks};
    std::vector<Mail> mail;
    std::vector<std::pair<Sakura::ToastType, std::string>> toasts;
    std::deque<std::function<void()>> commands;
    std::string systemClipboard;
    U64 delivered = 0;

    ClipboardFixture() {
        state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
        callbacks.enqueueMail = [this](Mail&& message) {
            mail.push_back(std::move(message));
        };
        callbacks.notify = [this](Sakura::ToastType type, I32, const std::string& message) {
            toasts.emplace_back(type, message);
        };
        callbacks.notifyResult = [](Result, const std::string&) {};
        callbacks.setClipboardText = [this](const std::string& text) {
            systemClipboard = text;
        };
        callbacks.getClipboardText = [this]() {
            return systemClipboard;
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
        callbacks.enqueueFlowgraphEdit = [this](FlowgraphEdit edit) {
            history.enqueueEdit(std::move(edit));
        };
        history.reconcile();
        drain();
    }

    ~ClipboardFixture() {
        state.flowgraph.history.clear();
    }

    void drain() {
        while (!commands.empty() || delivered < mail.size()) {
            if (!commands.empty()) {
                auto next = std::move(commands.front());
                commands.pop_front();
                next();
                continue;
            }
            const Mail message = mail[delivered++];
            std::visit([this](const auto& forwarded) {
                using Forwarded = std::decay_t<decltype(forwarded)>;
                if constexpr (std::is_same_v<Forwarded, MailSelectBlocks> ||
                              std::is_same_v<Forwarded, MailDeleteBlock>) {
                    REQUIRE(flowgraphActions.handle(forwarded) == Result::SUCCESS);
                }
            }, message);
        }
    }

    void copy(std::vector<std::string> blocks, bool cut = false) {
        REQUIRE(flowgraphActions.handle(MailCopyBlock{.flowgraph = "graph", .blockIds = std::move(blocks), .cut = cut}) ==
                Result::SUCCESS);
        drain();
    }

    void paste(std::optional<Extent2D<F32>> point) {
        REQUIRE(flowgraphActions.handle(MailPasteBlock{.flowgraph = "graph", .gridPosition = point}) ==
                Result::SUCCESS);
        drain();
    }

    void duplicate(std::vector<std::string> blocks) {
        REQUIRE(flowgraphActions.handle(MailDuplicateBlock{.flowgraph = "graph", .blockIds = std::move(blocks)}) ==
                Result::SUCCESS);
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

    FlowgraphSnapshot clipboard() const {
        return DecodeFlowgraphBlocks(systemClipboard).value_or(FlowgraphSnapshot{});
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

    std::vector<std::string> selected() {
        const auto selection = state.flowgraph.selections.find("graph");
        return selection == state.flowgraph.selections.end() ? std::vector<std::string>{} : selection->second.blocks;
    }

    void createChain() {
        TensorMap middle;
        middle["buffer"].requested("source", "signal");
        TensorMap leaf;
        leaf["buffer"].requested("middle", "buffer");
        REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("middle", TestFlowgraph::kSyntheticPassType, {}, middle) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("leaf", TestFlowgraph::kSyntheticPassType, {}, leaf) == Result::SUCCESS);
        setNode("source", {0.0f, 0.0f, 200.0f, 100.0f});
        setNode("middle", {300.0f, 40.0f, 220.0f, 100.0f});
        setNode("leaf", {600.0f, 80.0f, 240.0f, 100.0f});
        state.flowgraph.history.clear();
        history.reconcile();
        drain();
    }

    std::string sourceOf(const std::string& block) {
        const auto data = viewBlock(block);
        const auto input = data.inputs.find("buffer");
        if (input == data.inputs.end() || !input->second.external.has_value()) {
            return "";
        }
        return input->second.external->block;
    }

    void setNode(const std::string& block, NodeMeta meta) {
        REQUIRE(flowgraph->metadata().set("node", meta, block) == Result::SUCCESS);
    }

    NodeMeta node(const std::string& block) {
        NodeMeta meta;
        REQUIRE(flowgraph->metadata().get("node", meta, block) == Result::SUCCESS);
        return meta;
    }

    std::vector<std::string> removedTiles() {
        std::vector<std::string> removed;
        for (const auto& message : mail) {
            if (const auto* remove = std::get_if<MailRemoveLayoutBlock>(&message)) {
                removed.push_back(remove->block);
            }
        }
        std::sort(removed.begin(), removed.end());
        return removed;
    }
};

}  // namespace

TEST_CASE_METHOD(ClipboardFixture, "Copied blocks paste under new names with their shared links as one undo step",
                 "[core][compositor][clipboard]") {
    createChain();

    copy({"leaf", "middle"});
    REQUIRE(clipboard().blocks.size() == 2);
    CHECK(clipboard().find("middle")->inputs.empty());
    REQUIRE(clipboard().find("leaf")->inputs.contains("buffer"));
    CHECK(clipboard().find("leaf")->inputs.at("buffer").block == "middle");
    REQUIRE(toasts.size() == 1);
    CHECK(toasts.back().second == "2 blocks copied to clipboard.");
    CHECK_FALSE(graphHistory().canUndo());

    paste(Extent2D<F32>{1000.0f, 500.0f});
    CHECK(blocks() == std::vector<std::string>{"leaf", "leaf_1", "middle", "middle_1", "source"});
    CHECK(sourceOf("middle_1").empty());
    CHECK(sourceOf("leaf_1") == "middle_1");
    CHECK(sourceOf("leaf") == "middle");
    CHECK(node("middle_1").x == 1000.0f);
    CHECK(node("middle_1").y == 500.0f);
    CHECK(node("middle_1").width == 220.0f);
    CHECK(node("leaf_1").x == 1300.0f);
    CHECK(node("leaf_1").y == 540.0f);
    CHECK(selected() == std::vector<std::string>{"middle_1", "leaf_1"});
    CHECK(state.flowgraph.selections.at("graph").requests == 1);

    paste(Extent2D<F32>{0.0f, 600.0f});
    CHECK(blocks() == std::vector<std::string>{"leaf", "leaf_1", "leaf_2", "middle", "middle_1", "middle_2", "source"});
    CHECK(sourceOf("leaf_2") == "middle_2");
    CHECK(selected() == std::vector<std::string>{"middle_2", "leaf_2"});
    CHECK(state.flowgraph.selections.at("graph").requests == 2);

    undo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "leaf_1", "middle", "middle_1", "source"});
    undo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
    CHECK_FALSE(graphHistory().canUndo());

    redo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "leaf_1", "middle", "middle_1", "source"});
    CHECK(sourceOf("leaf_1") == "middle_1");
}

TEST_CASE_METHOD(ClipboardFixture, "Cut removes the blocks in one undo step and pasting them back keeps their names",
                 "[core][compositor][clipboard]") {
    createChain();

    copy({"middle", "leaf"}, true);
    CHECK(blocks() == std::vector<std::string>{"source"});
    CHECK(removedTiles() == std::vector<std::string>{"leaf", "middle"});
    REQUIRE_FALSE(toasts.empty());
    CHECK(toasts.back().second == "2 blocks cut to clipboard.");
    CHECK(clipboard().blocks.size() == 2);

    paste(Extent2D<F32>{300.0f, 40.0f});
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
    CHECK(sourceOf("leaf") == "middle");
    CHECK(sourceOf("middle").empty());
    CHECK(node("leaf").x == 600.0f);
    CHECK(node("leaf").y == 80.0f);

    undo();
    CHECK(blocks() == std::vector<std::string>{"source"});
    undo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
    CHECK(sourceOf("middle") == "source");
    CHECK_FALSE(graphHistory().canUndo());
}

TEST_CASE_METHOD(ClipboardFixture, "Paste and undo requested before a copy or cut finishes run after it",
                 "[core][compositor][clipboard]") {
    createChain();
    copy({"source"});

    REQUIRE(flowgraphActions.handle(MailCopyBlock{.flowgraph = "graph", .blockIds = {"leaf"}}) == Result::SUCCESS);
    REQUIRE(flowgraphActions.handle(MailPasteBlock{.flowgraph = "graph", .gridPosition = Extent2D<F32>{0.0f, 600.0f}}) ==
            Result::SUCCESS);
    drain();
    CHECK(blocks() == std::vector<std::string>{"leaf", "leaf_1", "middle", "source"});
    REQUIRE(clipboard().blocks.size() == 1);
    CHECK(clipboard().blocks.front().name == "leaf");

    REQUIRE(flowgraphActions.handle(MailCopyBlock{.flowgraph = "graph", .blockIds = {"leaf_1"}, .cut = true}) ==
            Result::SUCCESS);
    REQUIRE(history.handle(MailUndoFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
    drain();
    CHECK(blocks() == std::vector<std::string>{"leaf", "leaf_1", "middle", "source"});
    CHECK(graphHistory().canRedo());

    undo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
}

TEST_CASE_METHOD(ClipboardFixture, "Paste after a queued cut that finds nothing uses the cut before it",
                 "[core][compositor][clipboard]") {
    createChain();
    copy({"source"});

    REQUIRE(flowgraphActions.handle(MailCopyBlock{.flowgraph = "graph", .blockIds = {"leaf"}, .cut = true}) ==
            Result::SUCCESS);
    REQUIRE(flowgraphActions.handle(MailCopyBlock{.flowgraph = "graph", .blockIds = {"leaf"}, .cut = true}) ==
            Result::SUCCESS);
    REQUIRE(flowgraphActions.handle(MailPasteBlock{.flowgraph = "graph", .gridPosition = Extent2D<F32>{0.0f, 600.0f}}) ==
            Result::SUCCESS);
    drain();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
    CHECK(sourceOf("leaf").empty());
    REQUIRE(clipboard().blocks.size() == 1);
    CHECK(clipboard().blocks.front().name == "leaf");
}

TEST_CASE_METHOD(ClipboardFixture, "Duplicate offsets the copies, selects them, and leaves the clipboard alone",
                 "[core][compositor][clipboard]") {
    createChain();
    copy({"source"});
    toasts.clear();

    duplicate({"middle", "leaf"});
    CHECK(blocks() == std::vector<std::string>{"leaf", "leaf_1", "middle", "middle_1", "source"});
    CHECK(sourceOf("leaf_1") == "middle_1");
    CHECK(sourceOf("middle_1").empty());
    CHECK(node("middle_1").x == 340.0f);
    CHECK(node("middle_1").y == 80.0f);
    CHECK(node("leaf_1").x == 640.0f);
    CHECK(node("leaf_1").width == 240.0f);
    CHECK(selected() == std::vector<std::string>{"middle_1", "leaf_1"});
    REQUIRE(clipboard().blocks.size() == 1);
    CHECK(clipboard().blocks.front().name == "source");
    CHECK(toasts.empty());

    duplicate({"leaf_1"});
    CHECK(selected() == std::vector<std::string>{"leaf_2"});

    undo();
    undo();
    CHECK(blocks() == std::vector<std::string>{"leaf", "middle", "source"});
    CHECK_FALSE(graphHistory().canUndo());
}

TEST_CASE_METHOD(ClipboardFixture, "Pasted blocks keep their panel state but open attached",
                 "[core][compositor][clipboard]") {
    REQUIRE(flowgraph->blockCreate("plot", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("config", ConfigMeta{.collapsed = true, .detached = true}, "plot") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("surface_main", SurfaceMeta{.attachedHeight = 300, .detached = true}, "plot") ==
            Result::SUCCESS);

    duplicate({"plot"});
    REQUIRE(blocks() == std::vector<std::string>{"plot", "plot_1"});

    ConfigMeta config;
    REQUIRE(flowgraph->metadata().get("config", config, "plot_1") == Result::SUCCESS);
    CHECK(config.collapsed);
    CHECK_FALSE(config.detached);
    SurfaceMeta surface;
    REQUIRE(flowgraph->metadata().get("surface_main", surface, "plot_1") == Result::SUCCESS);
    CHECK(surface.attachedHeight == 300);
    CHECK_FALSE(surface.detached);
    CHECK_FALSE(flowgraph->metadata().has("node", "plot_1"));

    REQUIRE(flowgraph->metadata().get("config", config, "plot") == Result::SUCCESS);
    CHECK(config.detached);
}

TEST_CASE_METHOD(ClipboardFixture, "Copy of missing blocks and paste of an empty clipboard change nothing",
                 "[core][compositor][clipboard]") {
    REQUIRE(flowgraph->blockCreate("a", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);

    paste(Extent2D<F32>{0.0f, 0.0f});
    REQUIRE(toasts.size() == 1);
    CHECK(toasts.back().first == Sakura::ToastType::Warning);

    copy({"missing"}, true);
    CHECK(clipboard().blocks.empty());
    CHECK(toasts.back().first == Sakura::ToastType::Error);
    CHECK(blocks() == std::vector<std::string>{"a"});
    CHECK(state.flowgraph.selections.empty());
}

TEST_CASE_METHOD(ClipboardFixture, "Copied blocks reach the system clipboard as flowgraph YAML",
                 "[core][compositor][clipboard]") {
    createChain();

    copy({"middle", "leaf"});
    CHECK(systemClipboard.starts_with("---\n"));
    CHECK(systemClipboard.find("version: 2") != std::string::npos);
    CHECK(systemClipboard.find("${graph.middle.output.buffer}") != std::string::npos);
    CHECK(systemClipboard.find("${graph.source.output.signal}") == std::string::npos);

    const auto decoded = DecodeFlowgraphBlocks(systemClipboard);
    REQUIRE(decoded.has_value());
    REQUIRE(decoded->blocks.size() == 2);
    REQUIRE(decoded->find("middle"));
    CHECK(decoded->find("middle")->type == TestFlowgraph::kSyntheticPassType);
    CHECK(decoded->find("middle")->inputs.empty());
    REQUIRE(decoded->find("middle")->node.has_value());
    CHECK(decoded->find("middle")->node->x == 300.0f);
    CHECK(decoded->find("middle")->node->width == 220.0f);
    REQUIRE(decoded->find("leaf"));
    CHECK(decoded->find("leaf")->inputs.at("buffer").block == "middle");
    CHECK(decoded->find("leaf")->inputs.at("buffer").port == "buffer");

}

TEST_CASE_METHOD(ClipboardFixture, "Blocks from another app paste from the system clipboard and plain text pastes nothing",
                 "[core][compositor][clipboard]") {
    createChain();
    copy({"source"});

    systemClipboard = jst::fmt::format(
        "version: 2\n"
        "graph:\n"
        "  - name: feed\n"
        "    module: {}\n"
        "    meta:\n"
        "      node: {{x: 100, y: 200, width: 180, height: 90}}\n"
        "  - name: tap\n"
        "    module: {}\n"
        "    input:\n"
        "      buffer: ${{graph.feed.output.signal}}\n"
        "    meta:\n"
        "      node: {{x: 400, y: 200, width: 180, height: 90}}\n",
        TestFlowgraph::kSyntheticSourceType,
        TestFlowgraph::kSyntheticPassType);

    paste(Extent2D<F32>{1000.0f, 0.0f});
    CHECK(blocks() == std::vector<std::string>{"feed", "leaf", "middle", "source", "tap"});
    CHECK(sourceOf("tap") == "feed");
    CHECK(node("feed").x == 1000.0f);
    CHECK(node("tap").x == 1300.0f);
    CHECK(selected() == std::vector<std::string>{"feed", "tap"});

    toasts.clear();
    systemClipboard = "plain text";
    paste(Extent2D<F32>{0.0f, 600.0f});
    CHECK(blocks() == std::vector<std::string>{"feed", "leaf", "middle", "source", "tap"});
    REQUIRE(toasts.size() == 1);
    CHECK(toasts.back().first == Sakura::ToastType::Warning);
}

TEST_CASE("System clipboard text that is not a block list is ignored", "[core][compositor][clipboard]") {
    CHECK_FALSE(DecodeFlowgraphBlocks("").has_value());
    CHECK_FALSE(DecodeFlowgraphBlocks("hello").has_value());
    CHECK(DecodeFlowgraphBlocks("\"version\": 2\n\"graph\":\n  - name: a\n    module: m\n").has_value());
    CHECK_FALSE(DecodeFlowgraphBlocks("graph: 3\n").has_value());
    CHECK_FALSE(DecodeFlowgraphBlocks("version: 2\ngraph: []\n").has_value());
    CHECK_FALSE(DecodeFlowgraphBlocks("version: 1\ngraph:\n  - name: a\n    module: m\n").has_value());
    CHECK_FALSE(DecodeFlowgraphBlocks("version: 2\ngraph:\n  - name: a.b\n    module: m\n").has_value());
    CHECK_FALSE(DecodeFlowgraphBlocks("version: 2\ngraph:\n  - name: a\n").has_value());
    CHECK_FALSE(DecodeFlowgraphBlocks("version: 2\ngraph:\n  - name: a\n    module: m\n"
                                      "  - name: a\n    module: m\n").has_value());
    CHECK_FALSE(DecodeFlowgraphBlocks("version: 2\ngraph:\n  - name: a\n    module: m\n"
                                      "    input:\n      x: a.b\n").has_value());

    const auto frozen = DecodeFlowgraphBlocks("version: 2\ngraph:\n  - name: a\n    module: m\n    frozen: true\n");
    REQUIRE(frozen.has_value());
    CHECK(frozen->blocks.front().frozen);
    std::string yaml;
    REQUIRE(EncodeFlowgraphBlocks(*frozen, yaml) == Result::SUCCESS);
    CHECK(yaml.find("frozen: true") != std::string::npos);

    const auto outside = DecodeFlowgraphBlocks("version: 2\ngraph:\n  - name: a\n    module: m\n"
                                               "    input:\n      x: ${graph.gone.output.y}\n");
    REQUIRE(outside.has_value());
    REQUIRE(outside->blocks.size() == 1);
    CHECK(outside->blocks.front().inputs.empty());
}

TEST_CASE("Pasted block names keep free names and count up from a numeric suffix", "[core][compositor][clipboard]") {
    const std::unordered_set<std::string> names = {"fft", "fft_1", "fft_2048", "a_b", "x_", "_1"};
    const auto taken = [&](const std::string& name) {
        return names.contains(name);
    };
    CHECK(Naming::NextName("ifft", taken) == "ifft");
    CHECK(Naming::NextName("fft", taken) == "fft_2");
    CHECK(Naming::NextName("fft_1", taken) == "fft_2");
    CHECK(Naming::NextName("fft_2048", taken) == "fft_2049");
    CHECK(Naming::NextName("a_b", taken) == "a_b_1");
    CHECK(Naming::NextName("x_", taken) == "x__1");
    CHECK(Naming::NextName("_1", taken) == "_1_1");
}
