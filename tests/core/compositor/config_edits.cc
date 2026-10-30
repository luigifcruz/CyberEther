#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <memory>
#include <vector>

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
