#include "jetstream/config.hh"

#if defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include "agent_provider.hh"

#include <catch2/generators/catch_generators.hpp>
#include <nlohmann/json.hpp>

using namespace Jetstream;
using Json = nlohmann::json;

namespace {

using Kind = Agent::Provider::Error::Kind;

std::string Named(const char* type, Json event) {
    event["type"] = type;
    return std::string("event: ") + type + "\n" + AgentTest::Sse::Event(event);
}

std::string TextBlock(int index, const std::string& first, const std::string& rest = {}) {
    std::string stream = Named("content_block_start", {{"index", index},
                                                       {"content_block", {{"type", "text"}, {"text", first}}}});
    if (!rest.empty()) {
        stream += Named("content_block_delta", {{"index", index},
                                                {"delta", {{"type", "text_delta"}, {"text", rest}}}});
    }
    return stream + Named("content_block_stop", {{"index", index}});
}

std::string Start(const Json& usage = {{"input_tokens", 10}, {"output_tokens", 1}}) {
    return Named("message_start", {{"message", {{"id", "msg_1"}, {"role", "assistant"}, {"usage", usage}}}});
}

std::string Finish(const char* reason, int outputTokens = 20) {
    return Named("message_delta", {{"delta", {{"stop_reason", reason}}}, {"usage", {{"output_tokens", outputTokens}}}}) +
           Named("message_stop", {});
}

Agent::Provider::Config Config(const AgentTest::ProviderServer& provider) {
    auto config = provider.config(Agent::Provider::Kind::Anthropic);
    config.path = "/v1/messages";
    config.api = Agent::Provider::Api::Messages;
    return config;
}

}  // namespace

TEST_CASE("Anthropic Messages streams thinking and tool calls and replays them on the next turn",
          "[agent][providers][messages]") {
    AgentTest::ProviderServer provider;
    std::vector<Json> requests;
    std::vector<httplib::Headers> headers;
    provider.server.Post("/v1/messages", [&](const auto& request, auto& response) {
        requests.push_back(Json::parse(request.body));
        headers.push_back(request.headers);

        std::string stream;
        if (requests.size() == 1) {
            stream += Start({{"input_tokens", 10}, {"cache_read_input_tokens", 80},
                             {"cache_creation_input_tokens", 10}, {"output_tokens", 1}});
            stream += Named("content_block_start", {{"index", 0},
                                                    {"content_block", {{"type", "thinking"}, {"thinking", ""}}}});
            stream += Named("content_block_delta", {{"index", 0},
                                                    {"delta", {{"type", "thinking_delta"}, {"thinking", "Let me look"}}}});
            stream += Named("content_block_delta", {{"index", 0},
                                                    {"delta", {{"type", "signature_delta"}, {"signature", "sig-1"}}}});
            stream += Named("content_block_stop", {{"index", 0}});
            stream += TextBlock(1, "Check", "ing");
            stream += Named("content_block_start", {{"index", 2}, {"content_block", {
                {"type", "tool_use"}, {"id", "toolu_1"}, {"name", "inspect"}, {"input", Json::object()}}}});
            stream += Named("content_block_delta", {{"index", 2},
                                                    {"delta", {{"type", "input_json_delta"}, {"partial_json", "{\"id\""}}}});
            stream += Named("content_block_delta", {{"index", 2},
                                                    {"delta", {{"type", "input_json_delta"}, {"partial_json", ":1}"}}}});
            stream += Named("content_block_stop", {{"index", 2}});
            stream += Finish("tool_use");
        } else {
            stream += Start() + TextBlock(0, "Done") + Finish("end_turn");
        }
        response.set_content(stream, "text/event-stream");
    });
    provider.start();

    const Agent::Provider client(Config(provider));
    const std::vector<Agent::Provider::Function> tools{
        {"inspect", "Inspect a graph", R"({"type":"object","properties":{"id":{"type":"integer"}}})"},
    };
    const auto ask = [&](const std::vector<Agent::Provider::Message>& messages) {
        auto request = AgentTest::Request(messages, tools);
        request.maxOutputTokens = 4096;
        request.reasoningEffort = "high";
        return request;
    };

    std::string streamed;
    std::string thoughts;
    std::vector<std::string> announced;
    const Agent::Provider::Callbacks callbacks{
        .onToken = [&](const std::string& token) { streamed += token; },
        .onReasoning = [&](const std::string& thought, bool) { thoughts += thought; },
        .onToolCall = [&](const std::string& name) { announced.push_back(name); },
    };

    Agent::Provider::Response response;
    REQUIRE(client.stream(ask({{"system", "Be brief"}, {"user", "Inspect this"}}), callbacks, response) ==
            Result::SUCCESS);

    const auto calls = response.toolCalls;
    const auto text = response.text;
    const auto state = response.state;
    CHECK(text == "Checking");
    CHECK(streamed == text);
    CHECK(thoughts == "Let me look");
    CHECK(announced == std::vector<std::string>{"inspect"});
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "toolu_1");
    CHECK(calls[0].name == "inspect");
    CHECK(calls[0].arguments == R"({"id":1})");
    CHECK(response.usage.inputTokens == 100);
    CHECK(response.usage.cachedTokens == 80);
    CHECK(response.usage.cacheWriteTokens == 10);
    CHECK(response.usage.outputTokens == 20);
    REQUIRE(state);

    const std::vector<Agent::Provider::Message> history = {
        {"system", "Be brief"},
        {"user", "Inspect this"},
        {"assistant", text, calls, "", {}, state},
        {"tool", R"({"found":true})", {}, "toolu_1"},
        {"user", "Screenshot", {}, "", {"data:image/png;base64,AA=="}},
    };
    REQUIRE(client.stream(ask(history), {}, response) == Result::SUCCESS);
    CHECK(response.text == "Done");
    CHECK(response.toolCalls.empty());

    REQUIRE(requests.size() == 2);
    REQUIRE(headers.size() == 2);
    CHECK(headers[0].find("x-api-key")->second == AgentTest::ProviderServer::keyValue);
    CHECK(headers[0].find("anthropic-version")->second == "2026-10-21");
    CHECK(headers[0].find("Authorization") == headers[0].end());

    const auto& first = requests[0];
    CHECK(first["model"] == AgentTest::ProviderServer::modelValue);
    CHECK(first["stream"] == true);
    CHECK(first["max_tokens"] == 4096);
    CHECK(first["thinking"] == Json({{"type", "adaptive"}, {"display", "summarized"}}));
    CHECK(first["output_config"]["effort"] == "high");
    CHECK_FALSE(first.contains("tool_choice"));
    CHECK(first["system"][0]["text"] == "Be brief");
    CHECK(first["system"][0]["cache_control"]["type"] == "ephemeral");
    CHECK(first["tools"][0]["name"] == "inspect");
    CHECK(first["tools"][0]["input_schema"]["type"] == "object");
    CHECK(first["tools"][0]["input_schema"]["properties"]["id"]["type"] == "integer");
    REQUIRE(first["messages"].size() == 1);
    CHECK(first["messages"][0]["role"] == "user");
    CHECK(first["messages"][0]["content"][0]["text"] == "Inspect this");
    CHECK(first["messages"][0]["content"][0]["cache_control"]["type"] == "ephemeral");

    const auto& second = requests[1];
    CHECK(second["thinking"] == Json({{"type", "adaptive"}}));
    REQUIRE(second["messages"].size() == 3);
    CHECK(second["messages"][1]["role"] == "assistant");
    CHECK(second["messages"][1]["content"] == Json::array({
        {{"type", "thinking"}, {"thinking", "Let me look"}, {"signature", "sig-1"}},
        {{"type", "text"}, {"text", "Checking"}},
        {{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "inspect"}, {"input", {{"id", 1}}}},
    }));

    const auto& last = second["messages"][2];
    CHECK(last["role"] == "user");
    REQUIRE(last["content"].size() == 3);
    CHECK(last["content"][0]["type"] == "tool_result");
    CHECK(last["content"][0]["tool_use_id"] == "toolu_1");
    CHECK(last["content"][0]["content"] == R"({"found":true})");
    CHECK(last["content"][1] == Json({{"type", "text"}, {"text", "Screenshot"}}));
    CHECK(last["content"][2]["type"] == "image");
    CHECK(last["content"][2]["source"] == Json({{"type", "base64"}, {"media_type", "image/png"}, {"data", "AA=="}}));
    CHECK(last["content"][2]["cache_control"]["type"] == "ephemeral");
}

TEST_CASE("Anthropic Messages falls back to portable history when the state belongs to another model",
          "[agent][providers][messages]") {
    Json request;
    AgentTest::ProviderServer provider;
    provider.server.Post("/v1/messages", [&](const auto& input, auto& response) {
        request = Json::parse(input.body);
        response.set_content(Start() + TextBlock(0, "Done") + Finish("end_turn"), "text/event-stream");
    });
    provider.start();

    const auto config = Config(provider);
    const auto foreign = Json::array({{{"type", "thinking"}, {"thinking", "private"}, {"signature", "sig"}}});
    const auto state = std::make_shared<const Agent::Provider::State>(Agent::Provider::State{
        config.host, config.path, "another-model", foreign.dump(),
    });
    const std::vector<Agent::Provider::FunctionCall> calls{{"toolu_9", "inspect", R"({"id":9})"}};
    const std::vector<Agent::Provider::Message> history = {
        {"user", "First"},
        {"assistant", "Checking", calls, "", {}, state},
        {"tool", "{}", {}, "toolu_9"},
        {"user", "Next"},
    };

    auto ask = AgentTest::Request(history);
    ask.allowTools = false;
    const std::vector<Agent::Provider::Function> tools{{"inspect", "Inspect", R"({"type":"object"})"}};
    ask.tools = tools;

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(config).stream(ask, {}, response) == Result::SUCCESS);

    CHECK(request.dump().find("private") == std::string::npos);
    CHECK(request["tool_choice"] == Json({{"type", "none"}}));
    CHECK(request["max_tokens"] == 16384);
    REQUIRE(request["messages"].size() == 3);
    CHECK(request["messages"][1]["content"] == Json::array({
        {{"type", "text"}, {"text", "Checking"}},
        {{"type", "tool_use"}, {"id", "toolu_9"}, {"name", "inspect"}, {"input", {{"id", 9}}}},
    }));
    CHECK(request["messages"][2]["content"][0]["type"] == "tool_result");
    CHECK(request["messages"][2]["content"][1]["text"] == "Next");
}

TEST_CASE("Anthropic Messages reports truncated, refused, and failed streams", "[agent][providers][messages]") {
    const auto scenario = GENERATE("max_tokens", "refusal", "error");
    CAPTURE(scenario);

    AgentTest::ProviderServer provider;
    provider.server.Post("/v1/messages", [scenario](const auto&, auto& response) {
        std::string stream = Start() + TextBlock(0, "Partial");
        if (std::string_view(scenario) == "error") {
            stream += Named("error", {{"error", {{"type", "overloaded_error"}, {"message", "Overloaded"}}}});
        } else {
            stream += Finish(scenario);
        }
        response.set_content(stream, "text/event-stream");
    });
    provider.start();

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(Config(provider)).stream(AgentTest::Request({{"user", "Hello"}}), {}, response) ==
            Result::ERROR);

    CHECK(response.text == "Partial");
    if (std::string_view(scenario) == "max_tokens") {
        CHECK(response.error.kind == Kind::OutputLimit);
    } else if (std::string_view(scenario) == "refusal") {
        CHECK(response.error.kind == Kind::InvalidRequest);
    } else {
        CHECK(response.error.kind == Kind::Server);
        CHECK(response.error.code == "overloaded_error");
        CHECK(response.error.message.find("Overloaded") != std::string::npos);
    }
}

TEST_CASE("Anthropic Messages drops thinking options a model rejects", "[agent][providers][messages]") {
    std::vector<Json> requests;
    AgentTest::ProviderServer provider;
    provider.server.Post("/v1/messages", [&](const auto& input, auto& response) {
        requests.push_back(Json::parse(input.body));
        if (requests.size() == 1) {
            response.status = 400;
            response.set_content(R"({"type":"error","error":{"type":"invalid_request_error",)"
                                 R"("message":"thinking.type: adaptive is not supported on this model"}})",
                                 "application/json");
            return;
        }
        response.set_content(Start() + TextBlock(0, "Done") + Finish("end_turn"), "text/event-stream");
    });
    provider.start();

    auto ask = AgentTest::Request({{"user", "Hello"}});
    ask.model = "legacy-model";
    ask.reasoningEffort = "high";

    Agent::Provider::Response response;
    const auto onReasoning = [](const auto&, bool) {};
    const auto result = Agent::Provider(Config(provider)).stream(ask, {.onReasoning = onReasoning}, response);
    CAPTURE(response.error.message);
    REQUIRE(result == Result::SUCCESS);

    CHECK(response.text == "Done");
    REQUIRE(requests.size() == 2);
    CHECK(requests[0].contains("thinking"));
    CHECK(requests[0].contains("output_config"));
    CHECK_FALSE(requests[1].contains("thinking"));
    CHECK_FALSE(requests[1].contains("output_config"));
}

TEST_CASE("Anthropic model discovery reads display names", "[agent][providers][messages][models]") {
    AgentTest::ProviderServer provider;
    provider.server.Get("/models", [](const auto& request, auto& response) {
        CHECK(request.has_header("x-api-key"));
        response.set_content(R"({"data":[{"id":"claude-test","display_name":"Claude Test","type":"model"}],)"
                             R"("has_more":false})", "application/json");
    });
    provider.start();

    std::vector<Agent::Provider::Model> models;
    REQUIRE(Agent::Provider(Config(provider)).models(models) == Result::SUCCESS);

    REQUIRE(models.size() == 1);
    CHECK(models[0].id == "claude-test");
    CHECK(models[0].name == "Claude Test");
}

TEST_CASE("Anthropic Messages carries tool images inside the tool result", "[agent][providers][messages][images]") {
    AgentTest::ProviderServer provider;
    Json captured;
    provider.server.Post("/v1/messages", [&](const auto& request, auto& response) {
        captured = Json::parse(request.body);
        response.set_content(Start() + TextBlock(0, "Done") + Finish("end_turn"), "text/event-stream");
    });
    provider.start();

    const std::vector<Agent::Provider::FunctionCall> calls{{"toolu_1", "inspect", "{}"}, {"toolu_2", "inspect", "{}"}};
    const std::vector<Agent::Provider::Message> history{
        {"user", "Inspect this"},
        {"assistant", "", calls},
        {"tool", R"({"found":true})", {}, "toolu_1", {"data:image/png;base64,AA=="}, nullptr, true},
        {"tool", R"({"ok":true})", {}, "toolu_2"},
    };
    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(Config(provider)).stream(AgentTest::Request(history), {}, response) == Result::SUCCESS);

    const auto& results = captured["messages"].back()["content"];
    REQUIRE(results.size() == 2);
    CHECK(results[0]["type"] == "tool_result");
    CHECK(results[0]["tool_use_id"] == "toolu_1");
    CHECK(results[0]["is_error"] == true);
    REQUIRE(results[0]["content"].size() == 2);
    CHECK(results[0]["content"][0] == Json({{"type", "text"}, {"text", R"({"found":true})"}}));
    CHECK(results[0]["content"][1]["type"] == "image");
    CHECK(results[0]["content"][1]["source"] ==
          Json({{"type", "base64"}, {"media_type", "image/png"}, {"data", "AA=="}}));
    CHECK(results[1]["content"] == R"({"ok":true})");
}

#endif
