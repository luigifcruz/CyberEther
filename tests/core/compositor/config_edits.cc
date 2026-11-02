#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <memory>
#include <vector>

#include "compositor/default/actions/config_drafts.hh"
#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/presenters/flowgraph/config.hh"
#include "compositor/default/presenters/flowgraph/editor/base.hh"

using namespace Jetstream;

TEST_CASE("Inline and detached config edits preserve their completion callbacks",
          "[core][compositor][config][pending]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> messages;
    callbacks.enqueueMail = [&](Mail&& mail) { messages.push_back(std::move(mail)); };
    const PresenterContext context{state, callbacks};
    auto graph = std::make_shared<Flowgraph>();
    bool completed = false;
    Result result = Result::SUCCESS;
    const auto complete = [&](const Result value) {
        completed = true;
        result = value;
    };

    SECTION("inline editor") {
        const auto editor = FlowgraphEditorPresenter(context).build("graph", graph);
        REQUIRE(editor.onReconfigureBlock);
        editor.onReconfigureBlock("radio", {{"gain", 25.0f}}, true, complete);
    }
    SECTION("detached config") {
        Flowgraph::View::BlockData block;
        block.config = {{"gain", 50.0f}};
        block.interfaceConfigs.push_back({.name = "gain", .format = {{"type", "range"}}});
        const auto fields = BuildFlowgraphDetachedConfigFields(callbacks.enqueueMail,
            state.flowgraph, "graph", "radio", "detached", block);
        REQUIRE(fields.size() == 1);
        REQUIRE(fields[0].onApply);
        fields[0].onApply({{"gain", 25.0f}}, true, complete);
    }
    REQUIRE(messages.size() == 1);
    const auto& message = std::get<MailReconfigureBlock>(messages[0]);
    REQUIRE(message.flowgraph == "graph");
    REQUIRE(message.blockId == "radio");
    REQUIRE(message.silent);
    REQUIRE(Parser::Get<F32>(message.config, "gain") == 25.0f);
    REQUIRE_FALSE(completed);
    REQUIRE(message.onComplete);
    message.onComplete(Result::ERROR);
    REQUIRE(completed);
    REQUIRE(result == Result::ERROR);
}

TEST_CASE("Config edit failures are acknowledged when command completion is dispatched",
          "[core][compositor][config][pending]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::function<Result()> command;
    std::function<void(Result)> dispatch;
    callbacks.enqueueCommandWithCompletion = [&](std::function<Result()> next, bool,
                                                 std::function<void(Result)> complete) {
        command = std::move(next);
        dispatch = std::move(complete);
    };
    FlowgraphActions actions(state, callbacks);
    bool completed = false;
    Result result = Result::SUCCESS;
    const MailReconfigureBlock message{
        .flowgraph = "graph",
        .blockId = "missing",
        .config = {{"gain", 25.0f}},
        .silent = true,
        .onComplete = [&](const Result value) {
            completed = true;
            result = value;
        },
    };
    SECTION("block is unavailable") {
        state.flowgraph.items["graph"] = std::make_shared<Flowgraph>();
        REQUIRE(actions.handle(message) == Result::SUCCESS);
        REQUIRE(command);
        REQUIRE(dispatch);
        REQUIRE_FALSE(completed);
        const auto outcome = command();
        REQUIRE(outcome == Result::ERROR);
        REQUIRE_FALSE(completed);
        dispatch(outcome);
    }
    SECTION("flowgraph is unavailable") {
        REQUIRE(actions.handle(message) == Result::ERROR);
        REQUIRE_FALSE(command);
    }
    REQUIRE(completed);
    REQUIRE(result == Result::ERROR);
}

TEST_CASE("Inline and detached config copies share one draft until the committed value settles elsewhere",
          "[core][compositor][config][draft]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> messages;
    callbacks.enqueueMail = [&](Mail&& mail) { messages.push_back(std::move(mail)); };
    ConfigDraftActions drafts(state.flowgraph);
    auto now = ConfigDraftActions::Clock::now();
    const auto settle = [&] {
        drafts.refresh(now);
        now += ConfigDraftActions::SettleInterval;
        drafts.refresh(now);
    };

    Flowgraph::View::BlockData block;
    block.config = {{"code", std::string("print(1)")}};
    block.interfaceConfigs.push_back({.name = "code", .format = {{"type", std::string("python")}}});
    state.flowgraph.blocks["graph"].push_back({"script", block});
    const auto commit = [&](const std::string& code) {
        state.flowgraph.blocks["graph"].front().data.config = {{"code", code}};
    };

    const auto build = [&](const std::string& viewId) {
        auto fields = BuildFlowgraphConfigFields(viewId, state.flowgraph.blocks["graph"].front().data);
        ApplyConfigDrafts(fields, callbacks.enqueueMail, state.flowgraph, "graph", "script");
        REQUIRE(fields.size() == 1);
        return fields[0];
    };
    const auto dispatch = [&] {
        for (const auto& mail : messages) {
            REQUIRE(drafts.handle(std::get<MailSetConfigDraft>(mail)) == Result::SUCCESS);
        }
        messages.clear();
    };

    const auto inlineField = build("node");
    REQUIRE(inlineField.onDraft);
    CHECK_FALSE(inlineField.draft.has_value());
    inlineField.onDraft(std::string("print(2)"), "print(1)");
    dispatch();
    settle();
    CHECK(build("node").draft == "print(2)");
    CHECK(build("detached").draft == "print(2)");

    build("detached").onDraft(std::string("print(1)"), "print(1)");
    dispatch();
    CHECK_FALSE(build("node").draft.has_value());

    build("detached").onDraft(std::string("print(3)"), "print(1)");
    dispatch();
    commit("print(3)");
    drafts.refresh(now);
    CHECK_FALSE(build("node").draft.has_value());
    commit("print(1)");
    drafts.refresh(now);
    CHECK(build("node").draft == "print(3)");

    commit("print(3)");
    settle();
    CHECK(state.flowgraph.configDrafts.empty());

    commit("");
    build("node").onDraft(std::string("print(5)"), "");
    dispatch();
    state.flowgraph.blocks["graph"].front().data.config.clear();
    drafts.refresh(now);
    CHECK(state.flowgraph.configDrafts.size() == 1);
    settle();
    CHECK(state.flowgraph.configDrafts.empty());

    commit("print(3)");
    build("node").onDraft(std::string("print(4)"), "print(3)");
    dispatch();
    state.flowgraph.blocks["graph"].clear();
    settle();
    CHECK(state.flowgraph.configDrafts.empty());

    state.flowgraph.blocks["graph"].push_back({"script", block});
    commit("print(3)");
    build("node").onDraft(std::string("print(6)"), "print(3)");
    dispatch();
    state.flowgraph.blocks.erase("graph");
    drafts.refresh(now);
    CHECK(state.flowgraph.configDrafts.empty());
}
