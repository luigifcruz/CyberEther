#include <catch2/catch_test_macros.hpp>

#include "jetstream/agent/tools.hh"
#include "compositor/default/actions/agent.hh"
#include "compositor/default/support/agent/providers.hh"
#include "compositor/default/support/agent/runtime.hh"
#include "compositor/default/support/agent/executor.hh"
#include "compositor/default/model/callbacks.hh"
#include "compositor/default/model/state.hh"
#include "compositor/default/presenters/settings/intelligence_model.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/registry.hh"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include <nlohmann/json.hpp>

#if defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include "agent_provider.hh"
#endif

using namespace Jetstream;
using namespace std::chrono_literals;

namespace {

AgentExecutor::Layouts Bounds(const std::string& flowgraph, Jetstream::Rect bounds, Extent2D<F32> size) {
    AgentExecutor::Layouts layouts;
    layouts[flowgraph].nodes["source"] = {bounds, size, std::nullopt};
    return layouts;
}

void Measure(AgentExecutor& bridge, const AgentExecutor::Layouts& layouts) {
    for (const auto& [flowgraph, layout] : layouts) {
        bridge.layout(flowgraph, layout);
    }
    bridge.poll();
}

template<typename Done>
void PollBridge(AgentExecutor& bridge, const Done& done) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!done() && std::chrono::steady_clock::now() < deadline) {
        bridge.poll();
        std::this_thread::sleep_for(1ms);
    }
}

struct ShutdownOnFailure {
    AgentExecutor& bridge;
    bool done = false;

    ~ShutdownOnFailure() {
        if (!done) {
            bridge.shutdown();
        }
    }
};

}  // namespace

TEST_CASE("Unavailable voice leaves the active text conversation alone",
          "[agent][compositor][mode]") {
    DefaultCompositorState state;
    state.agent.settings.enabled = true;
    state.agent.settings.customProviders.clear();
    state.agent.settings.models.clear();
    state.agent.ui.visible = true;

    U64 connections = 0;
    std::string notice;
    DefaultCompositorCallbacks callbacks;
    callbacks.notify = [&](auto, auto, const std::string& message) {
        notice = message;
    };

    AgentRuntime runtime(state, callbacks, {.live = [&](auto, auto) {
        ++connections;
        return std::unique_ptr<Agent::Live>{};
    }});
    AgentActions actions(state, callbacks, runtime);

    REQUIRE(actions.handle(MailSetAgentVoiceVisible{true}) == Result::SUCCESS);
    CHECK(connections == 0);
    CHECK(state.agent.ui.visible);
    CHECK_FALSE(state.agent.ui.voiceVisible);
    CHECK_FALSE(notice.empty());
}

TEST_CASE("Cancelled compositor commands cannot apply queued graph mutations",
          "[agent][compositor][cancellation]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(graph->blockCreate("source", "signal_generator", {}, {}, DeviceType::CPU,
                                RuntimeType::NATIVE, "generic") == Result::SUCCESS);

    DefaultCompositorState state;
    state.flowgraph.items["focused"] = graph;
    state.interface.focusedFlowgraph = "focused";

    std::vector<std::function<Result()>> commands;
    DefaultCompositorCallbacks callbacks;
    callbacks.enqueueCommand = [&](std::function<Result()> command, bool) {
        commands.push_back(std::move(command));
    };

    AgentExecutor bridge(state, callbacks);
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    std::atomic<U64> cancellationChecks{0};
    auto execute = bridge.bind();
    auto result = std::async(std::launch::async, [&] {
        const auto cancel = [cancelled, &cancellationChecks] {
            ++cancellationChecks;
            return cancelled->load();
        };
        return execute("apply_flowgraph_patch",
                       R"({"operations":[{"op":"set_node_layout","name":"source","x":123}]})",
                       cancel);
    });

    struct Shutdown {
        AgentExecutor& bridge;

        ~Shutdown() {
            bridge.shutdown();
        }
    } shutdown{bridge};

    PollBridge(bridge, [&] {
        return !commands.empty();
    });
    REQUIRE(commands.size() == 1);

    cancelled->store(true);
    REQUIRE(result.wait_for(1s) == std::future_status::ready);
    REQUIRE(result.get().content.find("cancelled") != std::string::npos);

    const auto checks = cancellationChecks.load();
    REQUIRE(commands.front()() == Result::SUCCESS);
    CHECK(cancellationChecks.load() == checks);
    CHECK_FALSE(graph->metadata().has("node", "source"));

    REQUIRE(graph->blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Workbench model metadata reaches automatic output budgets without overriding user limits",
          "[agent][compositor][models]") {
    DefaultCompositorState state;
    auto& settings = state.agent.settings;
    settings.provider = "OpenCode Go";
    settings.model = "deepseek-v4.1-flash";

    Settings::Intelligence::Model model;
    model.provider = settings.provider;
    model.id = settings.model;
    model.contextLength = 1000000;
    model.outputLength = 384000;
    settings.models.push_back(model);

    auto config = AgentRuntime::Chat(state.agent, {});
    CHECK(config.maxOutputTokens == 0);
    CHECK(config.modelOutputTokens == 384000);
    CHECK(config.contextLength == 1000000);

    settings.models.front().maxOutputTokens = 8192;
    config = AgentRuntime::Chat(state.agent, {});
    CHECK(config.maxOutputTokens == 8192);
    CHECK(config.modelOutputTokens == 384000);
}

TEST_CASE("A tool executor retains its graph across focus changes and rejects closed targets",
          "[agent][compositor][target]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto first = std::make_shared<Flowgraph>();
    auto second = std::make_shared<Flowgraph>();
    for (const auto& graph : {first, second}) {
        REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        REQUIRE(graph->blockCreate("source", "signal_generator", {}, {}, DeviceType::CPU,
                                  RuntimeType::NATIVE, "generic") == Result::SUCCESS);
    }

    DefaultCompositorState state;
    state.flowgraph.items = {{"first", first}, {"second", second}};
    state.interface.focusedFlowgraph = "first";

    DefaultCompositorCallbacks callbacks;
    callbacks.enqueueCommand = [](std::function<Result()> command, bool) {
        command();
    };

    AgentExecutor bridge(state, callbacks);
    auto executor = bridge.bind();
    state.interface.focusedFlowgraph = "second";

    const auto invoke = [&] {
        auto future = std::async(std::launch::async, [&] {
            return executor("apply_flowgraph_patch",
                            R"({"operations":[{"op":"set_node_layout","name":"source","x":123}]})",
                            {});
        });

        PollBridge(bridge, [&] {
            return future.wait_for(0ms) == std::future_status::ready;
        });
        if (future.wait_for(0ms) != std::future_status::ready) {
            bridge.shutdown();
        }

        return nlohmann::json::parse(future.get().content);
    };

    CHECK(invoke()["status"] == "success");

    Flowgraph::Metadata::NodeLayout layout;
    REQUIRE(first->metadata().get("node", layout, "source") == Result::SUCCESS);
    CHECK(layout.x == 123);
    CHECK_FALSE(second->metadata().has("node", "source"));

    state.flowgraph.items.erase("first");
    CHECK(invoke().contains("error"));
    CHECK_FALSE(second->metadata().has("node", "source"));

    bridge.shutdown();
    for (const auto& graph : {first, second}) {
        REQUIRE(graph->blockDestroy("source", false) == Result::SUCCESS);
        REQUIRE(graph->destroy() == Result::SUCCESS);
    }
}

TEST_CASE("Tools bound while the Start, Settings or Docs tab is active never touch the hidden flowgraph",
          "[agent][compositor][target]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(graph->blockCreate("source", "signal_generator", {}, {}, DeviceType::CPU,
                              RuntimeType::NATIVE, "generic") == Result::SUCCESS);

    DefaultCompositorState state;
    state.flowgraph.items = {{"hidden", graph}};
    state.interface.focusedFlowgraph = "hidden";

    DefaultCompositorCallbacks callbacks;
    callbacks.enqueueCommand = [](std::function<Result()> command, bool) {
        command();
    };

    AgentExecutor bridge(state, callbacks);

    const auto invoke = [&](F32 x) {
        auto executor = bridge.bind();
        const auto args = jst::fmt::format(
            R"({{"operations":[{{"op":"set_node_layout","name":"source","x":{}}}]}})", x);
        auto future = std::async(std::launch::async, [&] {
            return executor("apply_flowgraph_patch", args, {});
        });

        PollBridge(bridge, [&] {
            return future.wait_for(0ms) == std::future_status::ready;
        });
        if (future.wait_for(0ms) != std::future_status::ready) {
            bridge.shutdown();
        }

        return nlohmann::json::parse(future.get().content);
    };

    state.interface.settingsTab = true;
    state.interface.focusedTab = WorkspaceTab::Settings;
    REQUIRE_FALSE(state.activeFlowgraph().has_value());
    CHECK(invoke(111).contains("error"));
    CHECK_FALSE(graph->metadata().has("node", "source"));

    state.interface.startTab = true;
    state.interface.focusedTab = WorkspaceTab::Start;
    REQUIRE_FALSE(state.activeFlowgraph().has_value());
    CHECK(invoke(222).contains("error"));
    CHECK_FALSE(graph->metadata().has("node", "source"));

    state.interface.docsTab = true;
    state.interface.focusedTab = WorkspaceTab::Docs;
    REQUIRE_FALSE(state.activeFlowgraph().has_value());
    CHECK(invoke(444).contains("error"));
    CHECK_FALSE(graph->metadata().has("node", "source"));

    state.interface.focusedTab.reset();
    REQUIRE(state.activeFlowgraph() == "hidden");
    CHECK(invoke(333)["status"] == "success");

    Flowgraph::Metadata::NodeLayout layout;
    REQUIRE(graph->metadata().get("node", layout, "source") == Result::SUCCESS);
    CHECK(layout.x == 333);

    bridge.shutdown();
    REQUIRE(graph->blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Malformed open_example arguments return errors from the UI poll",
          "[agent][compositor][validation]") {
    DefaultCompositorState state;
    U64 queuedCommands = 0;
    DefaultCompositorCallbacks callbacks;
    callbacks.enqueueCommand = [&](std::function<Result()>, bool) {
        ++queuedCommands;
    };

    AgentExecutor bridge(state, callbacks);
    auto execute = bridge.bind();

    const auto cases = {
        R"({"key":null})",
        R"({"key":42})",
        R"({"key":true})",
        R"({"key":[]})",
        R"({"key":{}})",
        R"({"key":""})",
        "{}",
        "[]",
        "null",
        "{",
        R"({"key":"missing-example"})",
    };

    for (const auto* args : cases) {
        CAPTURE(args);

        auto result = std::async(std::launch::async, [&] {
            return execute("open_example", args, {});
        });
        ShutdownOnFailure guard{bridge};

        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (result.wait_for(0ms) != std::future_status::ready &&
               std::chrono::steady_clock::now() < deadline) {
            REQUIRE_NOTHROW(bridge.poll());
            std::this_thread::sleep_for(1ms);
        }
        REQUIRE(result.wait_for(0ms) == std::future_status::ready);

        const auto reply = result.get();
        const auto body = nlohmann::json::parse(reply.content);
        REQUIRE(body.contains("error"));
        CHECK(body["error"].is_string());
        CHECK(reply.images.empty());
        CHECK(queuedCommands == 0);
        CHECK_FALSE(state.interface.pendingFocusedFlowgraph.has_value());

        guard.done = true;
    }
}

TEST_CASE("Agent tools return fresh rendered bounds without changing automatic sizing",
          "[agent][compositor][bounds]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);

    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    DefaultCompositorState state;
    state.flowgraph.items["focused"] = graph;
    state.interface.focusedFlowgraph = "focused";

    std::vector<std::function<Result()>> commands;
    DefaultCompositorCallbacks callbacks;
    callbacks.enqueueCommand = [&](std::function<Result()> command, bool) {
        commands.push_back(std::move(command));
    };

    AgentExecutor bridge(state, callbacks);

    const auto invoke = [&](const std::string& tool,
                            const std::string& args,
                            const std::function<AgentExecutor::Layouts()>& render) {
        auto execute = bridge.bind();
        auto result = std::async(std::launch::async, [&] {
            return execute(tool, args, {});
        });
        ShutdownOnFailure guard{bridge};

        PollBridge(bridge, [&] {
            return !commands.empty();
        });
        REQUIRE(commands.size() == 1);

        auto command = std::move(commands.front());
        commands.clear();
        REQUIRE(command() == Result::SUCCESS);

        Measure(bridge, Bounds("focused", {-100.0f, -100.0f, 999.0f, 999.0f}, {983.0f, 983.0f}));
        REQUIRE(result.wait_for(0ms) == std::future_status::timeout);

        for (U64 frame = 0; frame < 8 && result.wait_for(0ms) != std::future_status::ready; ++frame) {
            if (render) {
                Measure(bridge, render());
            } else {
                bridge.poll();
            }
        }
        REQUIRE(result.wait_for(1s) == std::future_status::ready);

        auto body = nlohmann::json::parse(result.get().content);
        guard.done = true;
        return body;
    };

    const auto render = [] {
        return Bounds("focused", {20.0f, 30.0f, 236.0f, 186.0f}, {220.0f, 170.0f});
    };
    const nlohmann::json expected = {
        {"x", 20.0f},
        {"y", 30.0f},
        {"width", 236.0f},
        {"height", 186.0f},
    };

    const auto created = invoke("apply_flowgraph_patch", R"({"operations":[
        {"op":"create_block","type":"signal_generator","name":"source","ref":"new"},
        {"op":"set_node_layout","name":"$new","x":20,"y":30}
    ]})", render);
    REQUIRE(created["status"] == "success");
    for (const auto& result : created["results"]) {
        CHECK(result["node"]["x"] == 20.0f);
        CHECK_FALSE(result["node"].contains("width"));
        CHECK_FALSE(result["node"].contains("height"));
        CHECK(result["layout_status"] == "measured");
        CHECK(result["bounds"] == expected);
    }

    const auto inspected = invoke("inspect_block", R"({"name":"source"})", render);
    CHECK(inspected["layout_status"] == "measured");
    CHECK(inspected["bounds"] == expected);

    for (const auto* detail : {"summary", "full", "routing"}) {
        const auto args = nlohmann::json({{"detail", detail}, {"sink", "source"}}).dump();
        const auto inspectedGraph = invoke("inspect_flowgraph", args, render);

        const bool routing = std::string(detail) == "routing";
        const auto& blocks = routing ? inspectedGraph["active_route"]["blocks"] : inspectedGraph["blocks"];
        CHECK(blocks[0]["bounds"] == expected);
    }

    const auto hidden = invoke("inspect_block", R"({"name":"source"})", {});
    CHECK(hidden["layout_status"] == "pending");
    CHECK(hidden["bounds"].is_null());

    const auto wrongGraph = invoke("inspect_block", R"({"name":"source"})", [] {
        return Bounds("other", {20.0f, 30.0f, 236.0f, 186.0f}, {220.0f, 170.0f});
    });
    CHECK(wrongGraph["layout_status"] == "pending");
    CHECK(wrongGraph["bounds"].is_null());

    const auto partial = invoke("apply_flowgraph_patch", R"({"operations":[
        {"op":"set_node_layout","name":"source","x":20},
        {"op":"delete_block","name":"missing"}
    ]})", render);
    CHECK(partial["status"] == "partial");
    CHECK(partial["results"][0]["bounds"] == expected);

    Flowgraph::Metadata::NodeLayout saved;
    REQUIRE(graph->metadata().get("node", saved, "source") == Result::SUCCESS);
    CHECK(saved.width == 0.0f);
    CHECK(saved.height == 0.0f);
    CHECK_FALSE(graph->metadata().has("bounds", "source"));

    REQUIRE(graph->blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Intelligence sub-pages open and focus the settings tab through the shared settings path",
          "[agent][compositor][settings]") {
    DefaultCompositorState state;
    state.agent.settings.enabled = false;
    state.interface.startTab = true;
    state.interface.focusedTab = WorkspaceTab::Start;
    state.settings.registryPage = RegistryPage::AddPlugin;
    DefaultCompositorCallbacks callbacks;
    AgentRuntime runtime(state, callbacks);
    AgentActions actions(state, callbacks, runtime);

    REQUIRE(actions.handle(MailOpenAgentProviderEditor{}) == Result::SUCCESS);
    CHECK(state.interface.settingsTab);
    CHECK(state.settingsTabActive());
    CHECK_FALSE(state.startTabActive());
    CHECK(state.interface.settingsTabRequests == 1);
    CHECK(state.settings.section == SettingsSection::Intelligence);
    CHECK(state.settings.intelligencePage == IntelligencePage::Provider);
    CHECK(state.settings.registryPage == RegistryPage::Overview);

    REQUIRE(actions.handle(MailOpenAgentModelPicker{}) == Result::SUCCESS);
    CHECK(state.interface.settingsTabRequests == 2);
    CHECK(state.settings.intelligencePage == IntelligencePage::AddModel);
}

TEST_CASE("A reply that finishes while the assistant is hidden stays unread until the chat is shown",
          "[agent][compositor][unread]") {
    DefaultCompositorState state;
    state.agent.settings.enabled = true;
    state.agent.settings.models.clear();
    DefaultCompositorCallbacks callbacks;
    AgentRuntime runtime(state, callbacks);
    AgentActions actions(state, callbacks, runtime);
    auto& conversation = state.agent.chat.conversation;
    const auto reply = [&](bool visible) {
        state.agent.ui.visible = visible;
        conversation.streaming = true;
        actions.reconcile();
        conversation.messages.push_back(std::make_shared<const Agent::Session::Entry>());
        conversation.streaming = false;
        actions.reconcile();
    };

    reply(true);
    CHECK_FALSE(state.agent.ui.unread);

    reply(false);
    CHECK(state.agent.ui.unread);
    actions.reconcile();
    CHECK(state.agent.ui.unread);

    REQUIRE(actions.handle(MailSetAgentChatVisible{.value = true}) == Result::SUCCESS);
    CHECK_FALSE(state.agent.ui.unread);

    reply(false);
    CHECK(state.agent.ui.unread);
    conversation.messages.clear();
    actions.reconcile();
    CHECK_FALSE(state.agent.ui.unread);
}

#if defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)

TEST_CASE("Model discovery failures reach the model picker with provider diagnostics",
          "[agent][compositor][models]") {
    AgentTest::ProviderServer provider;
    provider.server.Get("/models", [](const auto&, auto& response) {
        response.status = 503;
        response.set_content(R"({"error":{"message":"Catalog maintenance"}})", "application/json");
    });
    provider.start();

    DefaultCompositorState state;
    state.agent.settings.enabled = true;
    state.agent.settings.customProviders.push_back({
        .name = "Local",
        .host = provider.origin,
        .chatPath = "/chat",
        .modelsPath = "/models",
    });

    DefaultCompositorCallbacks callbacks;
    AgentRuntime runtime(state, callbacks);
    AgentActions actions(state, callbacks, runtime);

    actions.refresh();
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (state.agent.catalog.loading && std::chrono::steady_clock::now() < deadline) {
        actions.reconcile();
        std::this_thread::sleep_for(1ms);
    }
    REQUIRE_FALSE(state.agent.catalog.loading);

    PresenterContext context{state, callbacks};
    const auto config = IntelligenceModelPresenter(context).build();
    REQUIRE(config.warning);
    REQUIRE(config.status.find("HTTP 503") != std::string::npos);
    REQUIRE(config.status.find("Catalog maintenance") != std::string::npos);
    REQUIRE(config.status.find("API key") == std::string::npos);
}

TEST_CASE("Shutdown cancels in-flight model discovery and connection tests", "[agent][compositor][models]") {
    AgentTest::ProviderServer provider;
    std::atomic<int> entered{0};
    std::promise<void> release;
    auto released = release.get_future().share();
    provider.server.Get("/models", [&](const auto&, auto& response) {
        const auto provide = [&](size_t, httplib::DataSink& sink) {
            ++entered;
            released.wait();
            sink.done();
            return true;
        };
        response.set_chunked_content_provider("application/json", provide);
    });
    provider.start();

    const Settings::Intelligence::Provider local{
        .name = "Local",
        .host = provider.origin,
        .chatPath = "/chat",
        .modelsPath = "/models",
    };

    DefaultCompositorState state;
    state.agent.settings.enabled = true;
    state.agent.settings.customProviders.push_back(local);

    DefaultCompositorCallbacks callbacks;
    AgentRuntime runtime(state, callbacks);
    AgentActions actions(state, callbacks, runtime);

    actions.refresh();
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (entered.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    REQUIRE(entered.load() == 1);

    const auto start = std::chrono::steady_clock::now();
    actions.shutdown();
    REQUIRE(std::chrono::steady_clock::now() - start < 1s);
    REQUIRE_FALSE(state.agent.catalog.loading);
    REQUIRE(state.agent.catalog.models.empty());

    std::atomic<bool> abort{false};
    auto test = std::async(std::launch::async, [&] {
        return AgentProviders::Test(local, "", std::nullopt, [&] {
            return abort.load();
        });
    });
    while (entered.load() < 2 && std::chrono::steady_clock::now() < deadline + 3s) {
        std::this_thread::sleep_for(1ms);
    }
    REQUIRE(entered.load() == 2);
    abort.store(true);
    REQUIRE(test.wait_for(1s) == std::future_status::ready);
    REQUIRE_FALSE(test.get().first);

    release.set_value();
    provider.stop();
}

#endif
