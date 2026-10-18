#include <catch2/catch_test_macros.hpp>

#include "compositor/default/presenters/flowgraph/environment.hh"
#include "compositor/default/presenters/flowgraph/metadata.hh"
#include "flowgraph_fixture.hh"

#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace Jetstream;

namespace {

std::shared_ptr<Flowgraph> BorrowFixtureFlowgraph(Flowgraph& flowgraph) {
    return std::shared_ptr<Flowgraph>(&flowgraph, [](Flowgraph*) {});
}

struct KeyValueScene {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> mail;

    explicit KeyValueScene(Flowgraph& flowgraph) {
        state.flowgraph.items["graph"] = BorrowFixtureFlowgraph(flowgraph);
        state.interface.focusedFlowgraph = "graph";
        state.interface.flowgraphMetadataVisible = true;
        state.interface.flowgraphEnvironmentVisible = true;
        callbacks.enqueueMail = [this](Mail&& next) {
            mail.push_back(std::move(next));
        };
    }

    std::optional<FlowgraphKeyValueWindow::Config> metadata() const {
        const PresenterContext context{state, callbacks};
        return FlowgraphMetadataWindowPresenter(context).build();
    }

    std::optional<FlowgraphKeyValueWindow::Config> environment() const {
        const PresenterContext context{state, callbacks};
        return FlowgraphEnvironmentWindowPresenter(context).build();
    }
};

std::vector<std::string> Labels(const Sakura::Table::Nodes& nodes) {
    std::vector<std::string> labels;
    for (const auto& node : nodes) {
        labels.push_back(node.label);
    }
    return labels;
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture,
                 "Key-value trees keep metadata identities and labels distinct",
                 "[core][sakura][compositor][key-value]") {
    KeyValueScene scene(*flowgraph);

    const Parser::Map firstValue{{"child-A", "first value"}};
    const Parser::Map secondValue{{"child-B", "second value"}};
    bool environment = false;
    std::string firstLabel;
    std::string secondLabel;

    SECTION("slashes in block names and keys do not alias") {
        REQUIRE(flowgraph->blockCreate("a", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("a/b", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("b/c", firstValue, "a") == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("c", secondValue, "a/b") == Result::SUCCESS);
        firstLabel = "a.b/c";
        secondLabel = "a/b.c";
    }

    SECTION("flowgraph and block metadata may have identical display labels") {
        REQUIRE(flowgraph->blockCreate("a", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("a.metadata", firstValue) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("metadata", secondValue, "a") == Result::SUCCESS);
        firstLabel = secondLabel = "a.metadata";
    }

    SECTION("metadata labels containing id markers remain literal") {
        firstLabel = "alpha##literal###suffix";
        secondLabel = "beta##literal###suffix";
        REQUIRE(flowgraph->metadata().set(firstLabel, firstValue) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set(secondLabel, secondValue) == Result::SUCCESS);
    }

    SECTION("environment labels containing id markers remain literal") {
        environment = true;
        firstLabel = "alpha##literal###suffix";
        secondLabel = "beta##literal###suffix";
        REQUIRE(flowgraph->environment().set(firstLabel, firstValue) == Result::SUCCESS);
        REQUIRE(flowgraph->environment().set(secondLabel, secondValue) == Result::SUCCESS);
    }

    const auto initial = environment ? scene.environment() : scene.metadata();
    REQUIRE(initial.has_value());
    REQUIRE(initial->nodes.size() == 2);
    REQUIRE(initial->nodes[0].id != initial->nodes[1].id);
    REQUIRE(initial->nodes[0].label == firstLabel);
    REQUIRE(initial->nodes[1].label == secondLabel);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Key-value windows build only while visible for a focused open flowgraph",
                 "[core][sakura][compositor][key-value]") {
    KeyValueScene scene(*flowgraph);
    REQUIRE(scene.metadata().has_value());
    REQUIRE(scene.environment().has_value());

    SECTION("hidden windows") {
        scene.state.interface.flowgraphMetadataVisible = false;
        scene.state.interface.flowgraphEnvironmentVisible = false;
    }

    SECTION("no focused flowgraph") {
        scene.state.interface.focusedFlowgraph.reset();
    }

    SECTION("focused flowgraph that is no longer open") {
        scene.state.interface.focusedFlowgraph = "closed";
    }

    CHECK_FALSE(scene.metadata().has_value());
    CHECK_FALSE(scene.environment().has_value());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Metadata windows filter display keys case-insensitively and count shown against all entries",
                 "[core][sakura][compositor][key-value]") {
    KeyValueScene scene(*flowgraph);
    const Parser::Map value{{"child", std::string("value")}};
    REQUIRE(flowgraph->blockCreate("Radio", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("Gain", value) == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("alpha", value) == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("gain", value, "Radio") == Result::SUCCESS);

    const auto all = scene.metadata();
    REQUIRE(all.has_value());
    CHECK(Labels(all->nodes) == std::vector<std::string>{"Gain", "Radio.gain", "alpha"});
    CHECK(all->entryCount == "3 of 3 entries");
    CHECK(all->id == "flowgraph-metadata-window");
    CHECK(all->title == "Flowgraph Metadata (graph)");
    for (const auto& node : all->nodes) {
        CHECK(node.open);
    }

    scene.state.interface.flowgraphMetadataSearch = "GAIN";
    const auto filtered = scene.metadata();
    REQUIRE(filtered.has_value());
    CHECK(Labels(filtered->nodes) == std::vector<std::string>{"Gain", "Radio.gain"});
    CHECK(filtered->entryCount == "2 of 3 entries");
    CHECK(filtered->search == "GAIN");

    scene.state.interface.flowgraphMetadataSearch = "radio.";
    const auto byBlock = scene.metadata();
    REQUIRE(byBlock.has_value());
    CHECK(Labels(byBlock->nodes) == std::vector<std::string>{"Radio.gain"});
    CHECK(byBlock->entryCount == "1 of 3 entries");
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Environment windows sort keys and expand nested maps into child rows",
                 "[core][sakura][compositor][key-value]") {
    KeyValueScene scene(*flowgraph);
    REQUIRE(flowgraph->environment().set("zeta", Parser::Map{{"z", std::string("last")}}) == Result::SUCCESS);
    REQUIRE(flowgraph->environment().set("alpha",
                                         Parser::Map{{"a", std::string("one")}, {"b", std::string("two")}}) ==
            Result::SUCCESS);

    const auto window = scene.environment();
    REQUIRE(window.has_value());
    CHECK(window->id == "flowgraph-environment-window");
    CHECK(window->title == "Flowgraph Environment (graph)");
    CHECK(window->entryCount == "2 of 2 entries");
    REQUIRE(Labels(window->nodes) == std::vector<std::string>{"alpha", "zeta"});

    const auto& alpha = window->nodes[0];
    CHECK(alpha.open);
    CHECK(alpha.secondary);
    CHECK(alpha.cells == std::vector<std::string>{"2 keys"});
    CHECK(Labels(alpha.children) == std::vector<std::string>{"a", "b"});

    scene.state.interface.flowgraphEnvironmentSearch = "ZET";
    const auto filtered = scene.environment();
    REQUIRE(filtered.has_value());
    CHECK(Labels(filtered->nodes) == std::vector<std::string>{"zeta"});
    CHECK(filtered->entryCount == "1 of 2 entries");
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Key-value window search and close callbacks enqueue the matching mail",
                 "[core][sakura][compositor][key-value]") {
    KeyValueScene scene(*flowgraph);

    const auto metadata = scene.metadata();
    REQUIRE(metadata.has_value());
    metadata->onSearchChange("rate");
    metadata->onClose();

    const auto environment = scene.environment();
    REQUIRE(environment.has_value());
    environment->onSearchChange("path");
    environment->onClose();

    REQUIRE(scene.mail.size() == 4);
    REQUIRE(std::holds_alternative<MailSetFlowgraphMetadataSearch>(scene.mail[0]));
    CHECK(std::get<MailSetFlowgraphMetadataSearch>(scene.mail[0]).value == "rate");
    REQUIRE(std::holds_alternative<MailSetFlowgraphMetadataVisible>(scene.mail[1]));
    CHECK_FALSE(std::get<MailSetFlowgraphMetadataVisible>(scene.mail[1]).value);
    REQUIRE(std::holds_alternative<MailSetFlowgraphEnvironmentSearch>(scene.mail[2]));
    CHECK(std::get<MailSetFlowgraphEnvironmentSearch>(scene.mail[2]).value == "path");
    REQUIRE(std::holds_alternative<MailSetFlowgraphEnvironmentVisible>(scene.mail[3]));
    CHECK_FALSE(std::get<MailSetFlowgraphEnvironmentVisible>(scene.mail[3]).value);
}

TEST_CASE("Key-value nodes render scalars, vectors, and nested containers as text",
          "[core][sakura][compositor][key-value]") {
    using FlowgraphKeyValueDetail::AnyToNode;
    using FlowgraphKeyValueDetail::AnyToString;

    CHECK(AnyToString(std::any{}) == "null");
    CHECK(AnyToString(std::string("text")) == "text");
    CHECK(AnyToString(true) == "true");
    CHECK(AnyToString(I32{-7}) == "-7");
    CHECK(AnyToString(U8{200}) == "200");
    CHECK(AnyToString(std::vector<U64>{1, 2, 3}) == "[1, 2, 3]");
    CHECK(AnyToString(CF32{1.0f, -2.0f}) == "1-2i");
    CHECK(AnyToString(CF64{1.0, 2.0}) == "1+2i");
    CHECK(AnyToString(Parser::Map{{"k", std::string("v")}}) == "{k: v}");
    CHECK(AnyToString(Parser::Sequence{std::string("a"), true}) == "[a, true]");
    CHECK(AnyToString('x') == "?");

    const auto sequence = AnyToNode("list", "list", Parser::Sequence{std::string("a"), std::string("b")});
    CHECK(sequence.secondary);
    CHECK(sequence.cells == std::vector<std::string>{"2 items"});
    REQUIRE(sequence.children.size() == 2);
    CHECK(sequence.children[0].label == "[0]");
    CHECK(sequence.children[1].cells == std::vector<std::string>{"b"});

    const auto single = AnyToNode("map", "map", Parser::Map{{"only", U64{1}}});
    CHECK(single.cells == std::vector<std::string>{"1 key"});
    CHECK(single.children[0].cells == std::vector<std::string>{"1"});

    const auto leaf = AnyToNode("leaf", "Leaf", F32{2.5f});
    CHECK_FALSE(leaf.secondary);
    CHECK(leaf.children.empty());
    CHECK(leaf.cells == std::vector<std::string>{"2.5"});
}

TEST_CASE("Key-value entry counts pluralize by the total", "[core][sakura][compositor][key-value]") {
    CHECK(FlowgraphKeyValueDetail::EntryCount(0, 1) == "0 of 1 entry");
    CHECK(FlowgraphKeyValueDetail::EntryCount(1, 1) == "1 of 1 entry");
    CHECK(FlowgraphKeyValueDetail::EntryCount(2, 5) == "2 of 5 entries");
}

TEST_CASE("Key-value window views expose the configured floating window", "[core][sakura][compositor][key-value]") {
    FlowgraphKeyValueWindow view;
    bool closed = false;
    view.update({
        .id = "kv",
        .title = "Key Values",
        .search = "abc",
        .entryCount = "1 of 1 entry",
        .nodes = {{.id = "a", .label = "a", .cells = {"1"}}},
        .onClose = [&] { closed = true; },
    });

    const auto window = view.window();
    CHECK(window.id == "kv");
    CHECK(window.title == "Key Values");
    CHECK(window.content != nullptr);
    CHECK(window.size.x == 500.0f);
    CHECK(window.size.y == 700.0f);
    CHECK(window.padding == 8.0f);
    REQUIRE(window.onClose);
    window.onClose();
    CHECK(closed);
}
