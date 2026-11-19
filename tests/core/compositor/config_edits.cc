#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/presenters/flowgraph/config.hh"
#include "compositor/default/presenters/flowgraph/detached_config.hh"
#include "compositor/default/presenters/flowgraph/editor/base.hh"
#include "compositor/default/views/flowgraph/editor/config/advanced.hh"

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

TEST_CASE("Advanced settings stay behind a toggle that the node metadata remembers",
          "[core][compositor][config][advanced]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> messages;
    callbacks.enqueueMail = [&](Mail&& mail) { messages.push_back(std::move(mail)); };
    const PresenterContext context{state, callbacks};

    Flowgraph::View::BlockData block;
    block.config = {{"gain", 50.0f}, {"batches", U64{8}}};
    block.interfaceConfigs.push_back({.name = "batches",
                                      .format = {{"type", "uint"}, {"advanced", true}}});
    block.interfaceConfigs.push_back({.name = "gain", .format = {{"type", "range"}}});
    const auto names = [](const std::vector<FlowgraphConfigFieldConfig>& fields) {
        std::vector<std::string> result;
        for (const auto& field : fields) {
            result.push_back(field.name);
        }
        return result;
    };
    const auto built = BuildFlowgraphDetachedConfigFields(callbacks.enqueueMail,
        state.flowgraph, "graph", "radio", "detached", block);
    REQUIRE(names(built) == std::vector<std::string>{"batches", "gain"});

    std::vector<bool> toggles;
    const auto onToggle = [&](const bool advanced) { toggles.push_back(advanced); };

    auto hidden = built;
    ApplyAdvancedSection(hidden, "detached:advanced", false, onToggle);
    REQUIRE(names(hidden) == std::vector<std::string>{"gain", "advanced"});
    REQUIRE_FALSE(Parser::Get<bool>(hidden[1].values, "advanced"));
    REQUIRE(Parser::Get<std::string>(hidden[1].format, "type") == "bool");
    REQUIRE(Block::Interface::ValidateFormat(hidden[1].format) == Result::SUCCESS);

    bool completed = false;
    hidden[1].onApply({{"advanced", true}}, false, [&](const Result result) {
        completed = result == Result::SUCCESS;
    });
    REQUIRE(completed);
    REQUIRE(toggles == std::vector<bool>{true});
    REQUIRE(messages.empty());

    auto shown = built;
    ApplyAdvancedSection(shown, "detached:advanced", true, onToggle);
    REQUIRE(names(shown) == std::vector<std::string>{"gain", "advanced", "batches"});
    REQUIRE(Parser::Get<bool>(shown[1].values, "advanced"));
    shown[2].onApply({{"batches", U64{4}}}, false, {});
    REQUIRE(messages.size() == 1);
    REQUIRE(std::holds_alternative<MailReconfigureBlock>(messages[0]));

    auto plain = built;
    plain.erase(plain.begin());
    ApplyAdvancedSection(plain, "detached:advanced", false, onToggle);
    REQUIRE(names(plain) == std::vector<std::string>{"gain"});

    const auto editor = FlowgraphEditorPresenter(context).build("graph", std::make_shared<Flowgraph>());
    REQUIRE(editor.onNodeConfigAdvanced);
    editor.onNodeConfigAdvanced("radio", true);
    REQUIRE(messages.size() == 2);
    const auto& toggle = std::get<MailSetConfigAdvanced>(messages[1]);
    REQUIRE(toggle.flowgraph == "graph");
    REQUIRE(toggle.block == "radio");
    REQUIRE(toggle.advanced);

    const auto flowgraph = std::make_shared<Flowgraph>();
    state.flowgraph.blocks["graph"].push_back({"radio", block});
    REQUIRE(flowgraph->metadata().set("config", ConfigMeta{.detached = true, .advanced = true}, "radio") ==
            Result::SUCCESS);
    const auto detached = FlowgraphDetachedConfigPresenter(context).build("graph", flowgraph);
    REQUIRE(detached.size() == 1);
    REQUIRE(detached[0].configAdvanced);
    REQUIRE(detached[0].onConfigAdvanced);
    detached[0].onConfigAdvanced(false);
    REQUIRE(messages.size() == 3);
    const auto& detachedToggle = std::get<MailSetConfigAdvanced>(messages[2]);
    REQUIRE(detachedToggle.flowgraph == "graph");
    REQUIRE(detachedToggle.block == "radio");
    REQUIRE_FALSE(detachedToggle.advanced);
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
