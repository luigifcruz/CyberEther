#include <catch2/catch_test_macros.hpp>

#include "jetstream/agent/session.hh"
#include "jetstream/config.hh"

#if defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include <httplib.h>
#include <nlohmann/json.hpp>
#include "agent_provider.hh"
#include "agent_voice.hh"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <mutex>
#include <optional>
#include <thread>

using namespace Jetstream;
using namespace std::chrono_literals;

namespace {

struct LocalProvider : AgentTest::ProviderServer {
    std::mutex mutex;
    std::vector<nlohmann::json> requests;
    std::vector<std::string> sessionIds;
    std::atomic<bool> fail{false};

    LocalProvider() {
        server.Post("/chat", [this](const httplib::Request& request, httplib::Response& response) {
            size_t index;
            {
                std::lock_guard<std::mutex> lock(mutex);
                index = requests.size();
                requests.push_back(nlohmann::json::parse(request.body));
                sessionIds.push_back(request.get_header_value("x-opencode-session"));
            }

            if (fail.load()) {
                response.status = 500;
                response.set_content(R"({"error":{"message":"Inference server is overloaded"}})", "application/json");
                return;
            }

            if (index == 0) {
                response.set_content(
                    "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.summary\","
                    "\"summary\":\"**Inspecting**\\n\"}]}}]}\n\n"
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-one\","
                    "\"function\":{\"name\":\"inspect_tool\",\"arguments\":\"{\\\"x\\\":\"}}]}}]}\n\n"
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,"
                    "\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n"
                    "data: [DONE]\n\n",
                    "text/event-stream");
                return;
            }

            response.set_content(
                "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Ready to answer.\"}}]}\n\n"
                "data: {\"choices\":[{\"delta\":{\"content\":\"Done.\"}}]}\n\ndata: [DONE]\n\n",
                "text/event-stream");
        });
        start();
    }

    ~LocalProvider() {
        stop();
    }

    Agent::Session::Config config() const {
        return {
            .provider = std::make_shared<Agent::Provider>(ProviderServer::config(Agent::Provider::Kind::OpenCodeGo)),
            .model = modelValue,
            .systemContext = "Shared instructions for text and speech.",
            .tools = {{"inspect_tool", "Inspect a test value", R"({"type":"object"})"}},
        };
    }
};

bool WaitForCompletion(Agent::Session& session) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (session.snapshot().streaming) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

}  // namespace

TEST_CASE("Conversation sessions preserve tool exchanges, images, and follow-up context", "[agent][conversation]") {
    LocalProvider provider;
    Agent::Session session;

    auto config = provider.config();
    std::string toolName;
    std::string toolArgs;
    config.executeTool = [&](const std::string& name, const std::string& args, const Agent::Cancellation&) {
        toolName = name;
        toolArgs = args;
        return Agent::Tools::Output{R"({"value":42})", {"data:image/png;base64,dGVzdA=="}};
    };

    REQUIRE_FALSE(session.submit(" \n\t", config));
    REQUIRE(session.submit("Inspect this.", config));
    REQUIRE(WaitForCompletion(session));
    REQUIRE(toolName == "inspect_tool");
    REQUIRE(nlohmann::json::parse(toolArgs)["x"] == 1);

    const auto snapshot = session.snapshot();
    REQUIRE(snapshot.messages.size() == 2);
    REQUIRE(snapshot.messages[0].user);
    REQUIRE(snapshot.messages[1].segments.back().text == "Done.");
    REQUIRE(snapshot.messages[1].segments.front().kind == Agent::Session::Segment::Kind::Summary);

    REQUIRE(session.submit("Follow up.", config));
    REQUIRE(WaitForCompletion(session));

    std::lock_guard<std::mutex> lock(provider.mutex);
    REQUIRE(provider.requests.size() == 3);

    const auto& toolTurn = provider.requests[1]["messages"];
    REQUIRE(toolTurn[3]["role"] == "tool");
    REQUIRE(toolTurn[4]["content"][1]["image_url"]["url"] == "data:image/png;base64,dGVzdA==");

    const auto& followUp = provider.requests[2]["messages"];
    REQUIRE(followUp.size() == 7);
    REQUIRE(followUp[0]["content"] == config.systemContext);
    REQUIRE(followUp[1]["content"] == "Inspect this.");
    REQUIRE(followUp[2]["tool_calls"][0]["id"] == "call-one");
    REQUIRE(followUp[3]["role"] == "tool");
    REQUIRE(followUp[3]["tool_call_id"] == "call-one");
    for (size_t i = 0; i < toolTurn.size(); ++i) {
        CHECK(followUp[i] == toolTurn[i]);
    }
    REQUIRE(followUp[5]["content"] == "Done.");
    REQUIRE(followUp[5]["reasoning_content"] == "Ready to answer.");
    REQUIRE(followUp[6]["content"] == "Follow up.");

    REQUIRE_FALSE(provider.sessionIds[0].empty());
    REQUIRE(provider.sessionIds[0] == provider.sessionIds[2]);
}

TEST_CASE("Clearing a conversation cancels the active turn without reviving old messages", "[agent][conversation]") {
    LocalProvider provider;
    Agent::Session session;

    auto config = provider.config();
    std::promise<void> entered;
    std::promise<void> release;
    auto enteredFuture = entered.get_future();
    auto released = release.get_future().share();
    Agent::Cancellation turnCancelled;
    config.executeTool = [&](const std::string&, const std::string&, const Agent::Cancellation& shouldAbort) {
        turnCancelled = shouldAbort;
        entered.set_value();
        released.wait();
        return Agent::Tools::Output{"finished", {}};
    };

    REQUIRE(session.submit("Inspect this.", config));
    const bool reachedTool = enteredFuture.wait_for(5s) == std::future_status::ready;
    const bool rejectedConcurrent = !session.submit("Another turn.", config);
    session.clear();
    release.set_value();

    REQUIRE(reachedTool);
    REQUIRE(rejectedConcurrent);
    REQUIRE(WaitForCompletion(session));
    REQUIRE(session.snapshot().messages.empty());
    REQUIRE(session.snapshot().generation == 1);
    REQUIRE(turnCancelled());

    REQUIRE(session.submit("New conversation.", config));
    REQUIRE(turnCancelled());
    REQUIRE(WaitForCompletion(session));
    REQUIRE(session.snapshot().messages.size() == 2);

    std::lock_guard<std::mutex> lock(provider.mutex);
    REQUIRE(provider.requests.size() == 2);
    REQUIRE(provider.requests[1]["messages"].size() == 2);
    REQUIRE(provider.requests[1]["messages"][1]["content"] == "New conversation.");
}

TEST_CASE("Provider failures finish the turn with an error segment", "[agent][conversation]") {
    LocalProvider provider;
    provider.fail.store(true);

    Agent::Session session;
    const auto config = provider.config();
    REQUIRE(session.submit("Fail this request.", config));
    REQUIRE(WaitForCompletion(session));

    const auto snapshot = session.snapshot();
    const auto& last = snapshot.messages.back().segments.back();
    REQUIRE(last.kind == Agent::Session::Segment::Kind::Error);
    REQUIRE(last.text.find("Inference server is overloaded") != std::string::npos);
    REQUIRE(last.text.find("HTTP 500") != std::string::npos);

    provider.fail.store(false);
    REQUIRE(session.submit("Try again.", config));
    REQUIRE(WaitForCompletion(session));
    REQUIRE(session.snapshot().messages.back().segments.back().kind == Agent::Session::Segment::Kind::Text);

    std::lock_guard<std::mutex> lock(provider.mutex);
    for (const auto& message : provider.requests.back()["messages"]) {
        REQUIRE(message["content"].get<std::string>().find("overloaded") == std::string::npos);
    }
}

TEST_CASE("Changing between voice and text clears context", "[agent][conversation]") {
    LocalProvider provider;
    AgentTest::VoiceLink link;
    Agent::Session session(link.factory(), AgentTest::Silent);

    auto config = provider.config();
    config.executeTool = [](const std::string&, const std::string&, const Agent::Cancellation&) {
        return Agent::Tools::Output{"{}", {}};
    };

    REQUIRE(session.submit("Start in chat.", config));
    REQUIRE(WaitForCompletion(session));

    auto voice = config;
    voice.executeTool = [](const std::string&, const std::string&, const Agent::Cancellation&) {
        return Agent::Tools::Output{R"({"value":42})", {}};
    };
    REQUIRE(session.listen(voice));
    REQUIRE(session.snapshot().messages.empty());
    REQUIRE_FALSE(session.listen(voice));

    const auto callbacks = link.connection(0);
    REQUIRE(callbacks);
    callbacks->onInputTranscript("Change ");
    callbacks->onInputTranscript("the graph.");
    callbacks->onInputTranscriptDone();
    callbacks->onFunctionCalls({{"call-voice", "inspect_tool", "{}"}});
    REQUIRE(link.waitForOutputs(1));
    callbacks->onOutputTranscript("I changed it.");
    callbacks->onOutputTranscriptDone();
    session.hangup();

    REQUIRE(link.connections() == 1);
    REQUIRE(session.status().phase == Agent::Session::Phase::Idle);

    const auto transcript = session.snapshot();
    REQUIRE(transcript.messages.size() == 2);
    REQUIRE(transcript.messages[0].text == "Change the graph.");
    REQUIRE(transcript.messages[1].segments.front().text == "I changed it.");

    REQUIRE(link.outputs.size() == 1);
    REQUIRE(link.outputs[0].size() == 1);
    REQUIRE(link.outputs[0][0].first.id == "call-voice");
    REQUIRE(link.outputs[0][0].second.content == R"({"value":42})");

    REQUIRE(session.submit("Follow up in chat.", config));
    REQUIRE(WaitForCompletion(session));

    std::lock_guard<std::mutex> lock(provider.mutex);
    const auto& messages = provider.requests.back()["messages"];
    REQUIRE(messages.size() == 2);
    REQUIRE(messages[1]["content"] == "Follow up in chat.");
    CHECK(provider.sessionIds.front() != provider.sessionIds.back());
}

TEST_CASE("Clearing voice discards late transcripts and cancels tool execution", "[agent][conversation]") {
    AgentTest::VoiceLink link;
    Agent::Session session(link.factory(), AgentTest::Silent);
    session.listen({.systemContext = "instructions"});

    const auto callbacks = link.connection(0);
    REQUIRE(callbacks);
    callbacks->onInputTranscript("Old speech");
    session.clear();
    session.poll();
    REQUIRE(session.status().phase == Agent::Session::Phase::Idle);

    callbacks->onOutputTranscript("Late response");
    session.hangup();
    REQUIRE(session.snapshot().messages.empty());
}

TEST_CASE("Voice cancellation and transcript boundaries remain in the shared session", "[agent][conversation]") {
    AgentTest::VoiceLink link;
    Agent::Session session(link.factory(), AgentTest::Silent);
    session.listen({.systemContext = "instructions"});

    const auto callbacks = link.connection(0);
    REQUIRE(callbacks);
    callbacks->onInputTranscript("First utterance");
    callbacks->onInputTranscriptDone();
    callbacks->onInputTranscript("Second utterance");
    REQUIRE(session.snapshot().messages.size() == 2);

    session.cancel();
    session.poll();
    REQUIRE(session.status().phase == Agent::Session::Phase::Idle);

    callbacks->onOutputTranscript("Cancelled reply");
    session.hangup();
    REQUIRE(session.snapshot().messages.size() == 2);
}

TEST_CASE("Each voice connection starts fresh and stale callbacks are rejected",
          "[agent][conversation][history]") {
    AgentTest::VoiceLink link;
    Agent::Session session(link.factory(), AgentTest::Silent);

    session.listen({.systemContext = "seed"});
    const auto seed = link.connection(0);
    REQUIRE(seed);
    seed->onInputTranscript("Earlier question");
    seed->onInputTranscriptDone();
    seed->onOutputTranscript("Earlier answer");
    session.hangup();

    session.listen({.systemContext = "new instructions"});
    const auto voice = link.connection(1);
    REQUIRE(voice);
    voice->onInputTranscript("Speech after reservation");
    session.clear();
    session.poll();
    CHECK(session.status().phase == Agent::Session::Phase::Idle);
    CHECK(session.snapshot().messages.empty());

    session.listen({.systemContext = "current instructions"});
    const auto next = link.connection(2);
    REQUIRE(next);
    voice->onOutputTranscript("Stale reply");
    REQUIRE(session.status().phase != Agent::Session::Phase::Idle);

    next->onInputTranscript("Current speech");
    session.hangup();

    const auto transcript = session.snapshot();
    REQUIRE(transcript.messages.size() == 1);
    CHECK(transcript.messages[0].user);
    CHECK(transcript.messages[0].text == "Current speech");
    REQUIRE(link.configs.size() == 3);
    CHECK(link.configs[2].backendInstructions == "current instructions");
}

TEST_CASE("Text preparation payloads are released on the worker before publishing completion",
          "[agent][conversation][history]") {
    LocalProvider provider;
    Agent::Session session;

    std::promise<void> entered;
    std::promise<void> release;
    auto enteredFuture = entered.get_future();
    auto released = release.get_future().share();
    std::promise<std::pair<bool, std::thread::id>> destroyed;
    auto cleanup = destroyed.get_future();

    struct Payload {
        Agent::Session& session;
        std::promise<std::pair<bool, std::thread::id>>& destroyed;

        Payload(Agent::Session& session, decltype(destroyed) destroyed) : session(session), destroyed(destroyed) {}

        ~Payload() {
            destroyed.set_value({session.snapshot().streaming, std::this_thread::get_id()});
        }
    };

    auto payload = std::make_shared<Payload>(session, destroyed);
    auto config = provider.config();
    config.executeTool = [payload, released, &entered](const auto&, const auto&, const auto&) {
        entered.set_value();
        released.wait();
        return Agent::Tools::Output{"{}", {}};
    };

    REQUIRE(session.submit("Inspect before cleanup.", config));
    const bool reachedTool = enteredFuture.wait_for(5s) == std::future_status::ready;
    config.executeTool = {};
    payload.reset();
    release.set_value();
    const bool completed = WaitForCompletion(session);
    session.shutdown();

    REQUIRE(reachedTool);
    REQUIRE(completed);
    REQUIRE(cleanup.wait_for(0s) == std::future_status::ready);

    const auto [stillStreaming, thread] = cleanup.get();
    CHECK(stillStreaming);
    CHECK(thread != std::this_thread::get_id());
}

TEST_CASE("Responses final reasoning and phase survive the next user turn", "[agent][conversation][history]") {
    AgentTest::ProviderServer provider;
    std::vector<nlohmann::json> requests;

    const nlohmann::json reasoning = {
        {"type", "reasoning"},
        {"id", "rs-final"},
        {"summary", nlohmann::json::array()},
        {"encrypted_content", "opaque-final-state"},
    };
    const nlohmann::json message = {
        {"type", "message"},
        {"id", "msg-final"},
        {"role", "assistant"},
        {"phase", "final_answer"},
        {"status", "completed"},
        {"content", nlohmann::json::array({{{"type", "output_text"}, {"text", "Done"}}})},
    };
    const auto output = nlohmann::json::array({reasoning, message});

    provider.server.Post("/v1/responses", [&](const auto& request, auto& response) {
        requests.push_back(nlohmann::json::parse(request.body));

        const nlohmann::json completed = {{"type", "response.completed"}, {"response", {{"output", output}}}};
        response.set_content("data: " + completed.dump() + "\n\n", "text/event-stream");
    });
    provider.start();

    Agent::Session session;
    auto endpoint = provider.config(Agent::Provider::Kind::OpenAI);
    endpoint.path = "/v1/responses";
    endpoint.api = Agent::Provider::Api::Responses;
    Agent::Session::Config config{
        .provider = std::make_shared<Agent::Provider>(std::move(endpoint)),
        .model = AgentTest::ProviderServer::modelValue,
    };

    REQUIRE(session.submit("First question", config));
    REQUIRE(WaitForCompletion(session));

    REQUIRE(session.submit("Follow up", config));
    REQUIRE(WaitForCompletion(session));

    REQUIRE(requests.size() == 2);
    const auto& history = requests[1]["input"];
    REQUIRE(history.size() == 4);
    CHECK(history[0] == requests[0]["input"][0]);
    CHECK(history[1] == output[0]);
    CHECK(history[2] == output[1]);
    CHECK(history[3]["content"] == "Follow up");
    CHECK(requests[1]["prompt_cache_key"] == requests[0]["prompt_cache_key"]);
}

TEST_CASE("Switching models retains portable tool and image context without foreign reasoning",
          "[agent][conversation][history]") {
    LocalProvider provider;
    Agent::Session session;

    auto config = provider.config();
    config.executeTool = [](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{R"({"value":42})", {"data:image/png;base64,dGVzdA=="}};
    };

    REQUIRE(session.submit("Inspect", config));
    REQUIRE(WaitForCompletion(session));

    config.model = "another-model";
    REQUIRE(session.submit("Explain the image", config));
    REQUIRE(WaitForCompletion(session));

    std::lock_guard lock(provider.mutex);
    REQUIRE(provider.requests.size() == 3);

    const auto& messages = provider.requests.back()["messages"];
    REQUIRE(messages.size() == 7);
    CHECK(messages[2]["tool_calls"][0]["id"] == "call-one");
    CHECK(messages[3]["content"] == R"({"value":42})");
    CHECK(messages[4]["content"][1]["image_url"]["url"] == "data:image/png;base64,dGVzdA==");
    CHECK(messages[5]["content"] == "Done.");
    for (const auto& message : messages) {
        CHECK_FALSE(message.contains("reasoning_details"));
        CHECK_FALSE(message.contains("reasoning_content"));
    }
}

TEST_CASE("Cancelled tool batches replay completed results and close pending calls",
          "[agent][conversation][history]") {
    AgentTest::ProviderServer provider;
    std::vector<nlohmann::json> requests;
    provider.server.Post("/chat", [&](const auto& request, auto& response) {
        requests.push_back(nlohmann::json::parse(request.body));

        if (requests.size() == 1) {
            response.set_content(
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":["
                "{\"index\":0,\"id\":\"one\",\"function\":{\"name\":\"inspect\",\"arguments\":\"{}\"}},"
                "{\"index\":1,\"id\":\"two\",\"function\":{\"name\":\"inspect\",\"arguments\":\"{}\"}}"
                "]}}]}\n\ndata: [DONE]\n\n",
                "text/event-stream");
            return;
        }

        response.set_content("data: {\"choices\":[{\"delta\":{\"content\":\"Done\"}}]}\n\ndata: [DONE]\n\n",
                             "text/event-stream");
    });
    provider.start();

    Agent::Session session;
    int executed = 0;
    Agent::Session::Config config{
        .provider = std::make_shared<Agent::Provider>(provider.config()),
        .model = AgentTest::ProviderServer::modelValue,
        .tools = {{"inspect", "Inspect", R"({"type":"object"})"}},
        .executeTool = [&](const auto&, const auto&, const auto&) {
            ++executed;
            session.cancel();
            return Agent::Tools::Output{R"({"value":42})", {}};
        },
    };

    REQUIRE(session.submit("Inspect twice", config));
    REQUIRE(WaitForCompletion(session));
    REQUIRE(executed == 1);

    REQUIRE(session.submit("Continue", config));
    REQUIRE(WaitForCompletion(session));

    REQUIRE(requests.size() == 2);
    const auto& history = requests[1]["messages"];
    REQUIRE(history.size() == 5);
    CHECK(history[1]["tool_calls"].size() == 2);
    CHECK(history[2]["tool_call_id"] == "one");
    CHECK(history[2]["content"] == R"({"value":42})");
    CHECK(history[3]["tool_call_id"] == "two");
    CHECK(nlohmann::json::parse(history[3]["content"].get<std::string>()).contains("error"));
    CHECK(history[4]["content"] == "Continue");
}

#endif
