#include "jetstream/config.hh"

#if defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include "agent_provider.hh"

#include <catch2/generators/catch_generators.hpp>
#include <nlohmann/json.hpp>

using namespace Jetstream;
using Json = nlohmann::json;

namespace {

enum class Snapshot {
    Missing,
    Complete,
    Empty,
};

using AgentTest::Sse::Delta;
using AgentTest::Sse::Done;
using AgentTest::Sse::Event;

Json ReasoningItem(const char* id) {
    return {
        {"type", "reasoning"},
        {"id", id},
        {"summary", Json::array()},
        {"encrypted_content", std::string("opaque-") + id},
    };
}

}  // namespace

TEST_CASE("Reasoning replay preserves optional identities and streamed fields", "[agent][providers][replay]") {
    const bool indexed = GENERATE(false, true);
    const bool sharedId = GENERATE(false, true);

    Json thought{{"type", "reasoning.text"}, {"id", "rs-one"}, {"text", "First"}};
    Json encrypted{{"type", "reasoning.encrypted"}, {"id", sharedId ? "rs-one" : "rs-two"}, {"data", "opaque"}};
    if (indexed) {
        thought["index"] = 0;
        encrypted["index"] = 1;
    }

    auto continuation = thought;
    continuation["text"] = " second";
    continuation["signature"] = "sig-";

    auto signature = thought;
    signature.erase("text");
    signature["signature"] = "tail";

    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [&](const auto&, auto& response) {
        const std::string stream = Delta({{"reasoning_details", Json::array({thought, encrypted})}}) +
                                   Delta({{"reasoning_details", Json::array({continuation})}}) +
                                   Delta({{"reasoning_details", Json::array({signature})}}) +
                                   Delta({{"content", "Done"}}) +
                                   Done();
        response.set_content(stream, "text/event-stream");
    });
    provider.start();

    Agent::Provider::Response response;
    Agent::Provider client(provider.config());
    REQUIRE(client.stream(AgentTest::Request({{"user", "Hello"}}), {}, response) == Result::SUCCESS);

    const auto& state = response.state;
    REQUIRE(state);

    thought["text"] = "First second";
    thought["signature"] = "sig-tail";
    CHECK(Json::parse(state->payload)["reasoning_details"] == Json::array({thought, encrypted}));
}

TEST_CASE("Anonymous reasoning blocks keep their order across chunks", "[agent][providers][replay]") {
    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [](const auto&, auto& response) {
        const auto first = Json::array({
            {{"type", "reasoning.text"}, {"text", "One"}},
            {{"type", "reasoning.text"}, {"text", "Two"}},
        });
        const auto second = Json::array({
            {{"type", "reasoning.text"}, {"text", " A"}},
            {{"type", "reasoning.text"}, {"text", " B"}},
        });
        const std::string stream = Delta({{"reasoning_details", first}}) +
                                   Delta({{"reasoning_details", second}}) +
                                   Done();
        response.set_content(stream, "text/event-stream");
    });
    provider.start();

    Agent::Provider::Response response;
    Agent::Provider client(provider.config());
    REQUIRE(client.stream(AgentTest::Request({{"user", "Hello"}}), {}, response) == Result::SUCCESS);

    const auto& state = response.state;
    REQUIRE(state);

    const auto details = Json::parse(state->payload)["reasoning_details"];
    REQUIRE(details.size() == 2);
    CHECK(details[0]["text"] == "One A");
    CHECK(details[1]["text"] == "Two B");
}

TEST_CASE("Empty reasoning aliases do not hide populated reasoning content", "[agent][providers][replay]") {
    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [](const auto&, auto& response) {
        const std::string stream = Delta({{"reasoning", ""}, {"reasoning_content", "Keep this thought"}}) +
                                   Delta({{"content", "Done"}}) +
                                   Done();
        response.set_content(stream, "text/event-stream");
    });
    provider.start();

    Agent::Provider::Response response;
    Agent::Provider client(provider.config());
    REQUIRE(client.stream(AgentTest::Request({{"user", "Hello"}}), {}, response) == Result::SUCCESS);

    const auto& state = response.state;
    REQUIRE(state);
    CHECK(Json::parse(state->payload) == Json({{"reasoning_content", "Keep this thought"}}));
}

TEST_CASE("Responses replay retains ordered output including phase and encrypted reasoning",
          "[agent][providers][replay]") {
    const auto snapshot = GENERATE(Snapshot::Missing, Snapshot::Complete, Snapshot::Empty);

    const Json messageContent = Json::array({
        {{"type", "output_text"}, {"text", "Checking"}, {"annotations", Json::array()}},
    });
    const Json message = {
        {"type", "message"},
        {"id", "msg-one"},
        {"role", "assistant"},
        {"phase", "commentary"},
        {"status", "completed"},
        {"content", messageContent},
    };
    const Json call = {
        {"type", "function_call"},
        {"id", "fc-one"},
        {"call_id", "call-one"},
        {"name", "inspect"},
        {"arguments", "{}"},
    };
    const Json output = Json::array({ReasoningItem("rs-one"), message, ReasoningItem("rs-two"), call});

    std::vector<Json> requests;
    AgentTest::ProviderServer provider;
    provider.server.Post("/v1/responses", [&](const auto& request, auto& response) {
        requests.push_back(Json::parse(request.body));

        std::string stream = Event({{"type", "response.output_text.delta"}, {"delta", "Checking"}});
        for (const int i : {0, 3, 1, 2}) {
            auto item = output[i];
            if (snapshot == Snapshot::Complete && i == 1) {
                item["phase"] = "stale";
            }
            item["created_by"] = "output-only-provenance";
            stream += Event({{"type", "response.output_item.done"}, {"output_index", i}, {"item", item}});
        }

        Json completed{{"type", "response.completed"}};
        if (snapshot != Snapshot::Missing) {
            const Json finalOutput = snapshot == Snapshot::Complete ? output : Json::array();
            completed["response"] = {{"output", finalOutput}};
        }

        response.set_content(stream + Event(completed), "text/event-stream");
    });
    provider.start();

    auto config = provider.config(Agent::Provider::Kind::OpenAI);
    config.path = "/v1/responses";
    config.api = Agent::Provider::Api::Responses;
    Agent::Provider client(config);
    const auto ask = [](const std::vector<Agent::Provider::Message>& messages) {
        auto request = AgentTest::Request(messages);
        request.sessionId = "stable-session";
        return request;
    };

    Agent::Provider::Response response;
    REQUIRE(client.stream(ask({{"user", "Inspect"}}), {}, response) == Result::SUCCESS);

    const auto state = response.state;
    const auto calls = response.toolCalls;
    const auto text = response.text;
    REQUIRE(state);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "call-one");
    CHECK(text == "Checking");

    const std::vector<Agent::Provider::Message> history = {
        {"user", "Inspect"},
        {"assistant", text, calls, "", {}, state},
        {"tool", "{}", {}, "call-one"},
    };
    REQUIRE(client.stream(ask(history), {}, response) == Result::SUCCESS);

    REQUIRE(requests.size() == 2);
    CHECK(requests[0]["prompt_cache_key"] == "stable-session");
    CHECK(requests[1]["prompt_cache_key"] == requests[0]["prompt_cache_key"]);
    CHECK(requests[0]["store"] == false);

    const auto& replay = requests[1]["input"];
    REQUIRE(replay.size() == output.size() + 2);
    for (size_t i = 0; i < output.size(); ++i) {
        CHECK(replay[i + 1] == output[i]);
    }
    CHECK(replay.back()["type"] == "function_call_output");
}

TEST_CASE("Provider state is scoped to its endpoint and model", "[agent][providers][replay]") {
    const auto changed = GENERATE("host", "path", "model");
    const bool responses = GENERATE(false, true);

    Json request;
    AgentTest::ProviderServer provider;
    provider.server.Post(responses ? "/v1/responses" : "/chat", [&](const auto& input, auto& response) {
        request = Json::parse(input.body);

        std::string stream;
        if (responses) {
            stream = Event({{"type", "response.completed"}});
        } else {
            stream = Delta({{"content", "Done"}}) + Done();
        }
        response.set_content(stream, "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    if (responses) {
        config.path = "/v1/responses";
        config.api = Agent::Provider::Api::Responses;
    }

    std::string payload = R"({"reasoning_content":"private"})";
    if (responses) {
        payload = Json::array({ReasoningItem("private")}).dump();
    }

    Agent::Provider::State saved{config.host, config.path, AgentTest::ProviderServer::modelValue, payload};
    if (std::string_view(changed) == "host") {
        saved.host += "/different";
    }
    if (std::string_view(changed) == "path") {
        saved.path += "/different";
    }
    if (std::string_view(changed) == "model") {
        saved.model += "-different";
    }

    const auto state = std::make_shared<const Agent::Provider::State>(std::move(saved));
    const std::vector<Agent::Provider::Message> history = {
        {"user", "Hello"},
        {"assistant", "Portable answer", {}, "", {}, state},
        {"user", "Next"},
    };

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(config).stream(AgentTest::Request(history), {}, response) == Result::SUCCESS);

    const char* field = responses ? "input" : "messages";
    CHECK(request[field][1] == Json({{"role", "assistant"}, {"content", "Portable answer"}}));
    CHECK(request.dump().find("private") == std::string::npos);
}

TEST_CASE("Responses respect the visible reasoning preference while retaining continuation state",
          "[agent][providers][replay]") {
    Json request;
    AgentTest::ProviderServer provider;
    provider.server.Post("/v1/responses", [&](const auto& input, auto& response) {
        request = Json::parse(input.body);
        response.set_content(Event({{"type", "response.completed"}}), "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    config.path = "/v1/responses";
    config.api = Agent::Provider::Api::Responses;
    config.requestReasoning = false;

    Agent::Provider::Response response;
    const auto onReasoning = [](const auto&, bool) {};
    Agent::Provider client(config);
    REQUIRE(client.stream(AgentTest::Request({{"user", "Hello"}}), {.onReasoning = onReasoning}, response) ==
            Result::SUCCESS);

    CHECK_FALSE(request.contains("reasoning"));
    CHECK(request["include"] == Json::array({"reasoning.encrypted_content"}));
}

TEST_CASE("Unusable Responses state falls back to portable content and skips empty replies",
          "[agent][providers][replay]") {
    const bool portable = GENERATE(false, true);
    const bool modelSwitch = GENERATE(false, true);

    Json request;
    AgentTest::ProviderServer provider;
    provider.server.Post("/v1/responses", [&](const auto& input, auto& response) {
        request = Json::parse(input.body);
        response.set_content(Event({{"type", "response.completed"}}), "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    config.path = "/v1/responses";
    config.api = Agent::Provider::Api::Responses;

    const auto unusable = Json::array({{{"type", "reasoning"}, {"id", "rs-only"}, {"summary", Json::array()}}});
    const std::string model = modelSwitch ? "another-model" : AgentTest::ProviderServer::modelValue;
    auto state = std::make_shared<const Agent::Provider::State>(Agent::Provider::State{
        config.host,
        config.path,
        model,
        unusable.dump(),
    });

    std::vector<Agent::Provider::FunctionCall> calls;
    if (portable) {
        calls = {{"call-one", "inspect", "{}"}};
    }

    const std::string reply = portable ? "Checking" : "";
    std::vector<Agent::Provider::Message> history{{"user", "First"}, {"assistant", reply, calls, "", {}, state}};
    if (portable) {
        history.push_back({"tool", "{}", {}, "call-one"});
    }
    history.push_back({"user", "Next"});

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(config).stream(AgentTest::Request(history), {}, response) == Result::SUCCESS);

    const auto& input = request["input"];
    REQUIRE(input.size() == (portable ? 5 : 2));
    CHECK(input.front()["content"] == "First");
    CHECK(input.back()["content"] == "Next");
    if (portable) {
        CHECK(input[1]["content"] == "Checking");
        CHECK(input[2]["type"] == "function_call");
        CHECK(input[2]["call_id"] == input[3]["call_id"]);
    }
}

TEST_CASE("OpenRouter Claude requests mark cache boundaries and retain session affinity",
          "[agent][providers][cache]") {
    Json request;
    std::string session;
    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [&](const auto& input, auto& response) {
        request = Json::parse(input.body);
        session = input.get_header_value("x-session-id");
        response.set_content(Delta({{"content", "Done"}}) + Done(), "text/event-stream");
    });
    provider.start();

    const auto config = provider.config(Agent::Provider::Kind::OpenRouter);

    const std::vector<Agent::Provider::Message> history = {
        {"system", "Instructions"},
        {"user", "First"},
        {"assistant", "Earlier"},
        {"user", "Next"},
    };

    auto ask = AgentTest::Request(history);
    ask.model = "anthropic/claude-opus-5.5";
    ask.sessionId = "conversation-one";

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(config).stream(ask, {}, response) == Result::SUCCESS);

    CHECK(session == ask.sessionId);

    const auto& messages = request["messages"];
    CHECK(messages.front()["content"][0]["cache_control"]["type"] == "ephemeral");
    CHECK(messages.back()["content"][0]["cache_control"]["type"] == "ephemeral");
    CHECK(messages[1]["content"] == "First");
    CHECK(messages[2]["content"] == "Earlier");
}

#endif
