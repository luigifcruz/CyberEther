#include <catch2/catch_test_macros.hpp>

#include "compositor/default/presenters/flowgraph/environment.hh"
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
    std::shared_ptr<Flowgraph> flowgraph;

    explicit KeyValueScene(Flowgraph& flowgraph) : flowgraph(BorrowFixtureFlowgraph(flowgraph)) {
        state.flowgraph.items["graph"] = this->flowgraph;
        state.flowgraph.environment["graph"] = "";
        callbacks.enqueueMail = [this](Mail&& next) {
            mail.push_back(std::move(next));
        };
    }

    std::optional<FlowgraphEnvironment::Config> environment() const {
        const PresenterContext context{state, callbacks};
        return FlowgraphEnvironmentPresenter(context).build("graph", flowgraph);
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
                 "Environment labels containing id markers stay literal and distinct",
                 "[core][sakura][compositor][key-value]") {
    KeyValueScene scene(*flowgraph);
    const Parser::Map firstValue{{"child-A", "first value"}};
    const Parser::Map secondValue{{"child-B", "second value"}};
    const std::string firstLabel = "alpha##literal###suffix";
    const std::string secondLabel = "beta##literal###suffix";
    REQUIRE(flowgraph->environment().set(firstLabel, firstValue) == Result::SUCCESS);
    REQUIRE(flowgraph->environment().set(secondLabel, secondValue) == Result::SUCCESS);

    const auto initial = scene.environment();
    REQUIRE(initial.has_value());
    REQUIRE(initial->nodes.size() == 2);
    REQUIRE(initial->nodes[0].id != initial->nodes[1].id);
    REQUIRE(initial->nodes[0].label == firstLabel);
    REQUIRE(initial->nodes[1].label == secondLabel);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Environment tiles build only while open for their flowgraph",
                 "[core][sakura][compositor][key-value]") {
    KeyValueScene scene(*flowgraph);
    REQUIRE(scene.environment().has_value());

    scene.state.flowgraph.environment.erase("graph");
    CHECK_FALSE(scene.environment().has_value());

    scene.state.flowgraph.environment["other"] = "";
    CHECK_FALSE(scene.environment().has_value());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Environment tiles sort keys and expand nested maps into child rows",
                 "[core][sakura][compositor][key-value]") {
    KeyValueScene scene(*flowgraph);
    REQUIRE(flowgraph->environment().set("zeta", Parser::Map{{"z", std::string("last")}}) == Result::SUCCESS);
    REQUIRE(flowgraph->environment().set("alpha",
                                         Parser::Map{{"a", std::string("one")}, {"b", std::string("two")}}) ==
            Result::SUCCESS);

    const auto tile = scene.environment();
    REQUIRE(tile.has_value());
    CHECK(tile->id == kLayoutEnvironmentPane);
    CHECK(tile->title == "Environment");
    CHECK(tile->entryCount == "2 of 2 entries");
    REQUIRE(Labels(tile->nodes) == std::vector<std::string>{"alpha", "zeta"});

    const auto& alpha = tile->nodes[0];
    CHECK(alpha.open);
    CHECK(alpha.secondary);
    CHECK(alpha.cells == std::vector<std::string>{"2 keys"});
    CHECK(Labels(alpha.children) == std::vector<std::string>{"a", "b"});

    scene.state.flowgraph.environment["graph"] = "ZET";
    const auto filtered = scene.environment();
    REQUIRE(filtered.has_value());
    CHECK(Labels(filtered->nodes) == std::vector<std::string>{"zeta"});
    CHECK(filtered->entryCount == "1 of 2 entries");
    CHECK(filtered->search == "ZET");
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Environment tile search and close callbacks enqueue mails for their flowgraph",
                 "[core][sakura][compositor][key-value]") {
    KeyValueScene scene(*flowgraph);

    const auto environment = scene.environment();
    REQUIRE(environment.has_value());
    environment->onSearchChange("path");
    environment->onClose();

    REQUIRE(scene.mail.size() == 2);
    REQUIRE(std::holds_alternative<MailSetFlowgraphEnvironmentSearch>(scene.mail[0]));
    CHECK(std::get<MailSetFlowgraphEnvironmentSearch>(scene.mail[0]).flowgraph == "graph");
    CHECK(std::get<MailSetFlowgraphEnvironmentSearch>(scene.mail[0]).value == "path");
    REQUIRE(std::holds_alternative<MailSetFlowgraphEnvironmentVisible>(scene.mail[1]));
    CHECK(std::get<MailSetFlowgraphEnvironmentVisible>(scene.mail[1]).flowgraph == "graph");
    CHECK_FALSE(std::get<MailSetFlowgraphEnvironmentVisible>(scene.mail[1]).value);
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

TEST_CASE("Environment tile views expose their identity, title and close callback",
          "[core][sakura][compositor][key-value]") {
    FlowgraphEnvironment view;
    bool closed = false;
    view.update({
        .id = "environment",
        .title = "Environment",
        .search = "abc",
        .entryCount = "1 of 1 entry",
        .nodes = {{.id = "a", .label = "a", .cells = {"1"}}},
        .onClose = [&] { closed = true; },
    });

    CHECK(view.id() == "environment");
    CHECK(view.title() == "Environment");
    const auto close = view.closeCallback();
    REQUIRE(close);
    close();
    CHECK(closed);
}
