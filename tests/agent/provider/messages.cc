#include "jetstream/config.hh"

#if defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include "agent_provider.hh"

#include <catch2/generators/catch_generators.hpp>
#include <nlohmann/json.hpp>

#include <functional>
#include <utility>

using namespace Jetstream;
using Json = nlohmann::json;

namespace {

using Kind = Agent::Provider::Error::Kind;
using Message = Agent::Provider::Message;

std::string Event(const Json& event) {
    return "event: " + event["type"].get<std::string>() + "\ndata: " + event.dump() + "\n\n";
}

std::string Start(const Json& usage = {{"input_tokens", 1}, {"output_tokens", 1}}) {
    return Event({{"type", "message_start"}, {"message", {{"id", "msg_1"}, {"role", "assistant"}, {"usage", usage}}}});
}

std::string BlockStart(int index, const Json& block) {
    return Event({{"type", "content_block_start"}, {"index", index}, {"content_block", block}});
}

std::string BlockDelta(int index, const Json& delta) {
    return Event({{"type", "content_block_delta"}, {"index", index}, {"delta", delta}});
}

std::string BlockStop(int index) {
    return Event({{"type", "content_block_stop"}, {"index", index}});
}

std::string Stop(const char* reason, const Json& usage = {{"output_tokens", 1}}) {
    return Event({{"type", "message_delta"}, {"delta", {{"stop_reason", reason}}}, {"usage", usage}}) +
           Event({{"type", "message_stop"}});
}

std::string TextReply(const char* text) {
    return Start() +
           BlockStart(0, {{"type", "text"}, {"text", ""}}) +
           BlockDelta(0, {{"type", "text_delta"}, {"text", text}}) +
           BlockStop(0) +
           Stop("end_turn");
}

std::string ToolReply() {
    return Start({{"input_tokens", 10}, {"cache_read_input_tokens", 80}, {"cache_creation_input_tokens", 5}}) +
           Event({{"type", "ping"}}) +
           BlockStart(0, {{"type", "thinking"}, {"thinking", ""}, {"signature", ""}}) +
           BlockDelta(0, {{"type", "thinking_delta"}, {"thinking", "Look at "}}) +
           BlockDelta(0, {{"type", "thinking_delta"}, {"thinking", "the graph."}}) +
           BlockDelta(0, {{"type", "signature_delta"}, {"signature", "sig-one"}}) +
           BlockStop(0) +
           BlockStart(1, {{"type", "text"}, {"text", ""}}) +
           BlockDelta(1, {{"type", "text_delta"}, {"text", "Checking"}}) +
           BlockDelta(1, {{"type", "text_delta"}, {"text", " now"}}) +
           BlockStop(1) +
           BlockStart(2, {{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "inspect"}, {"input", Json::object()}}) +
           BlockDelta(2, {{"type", "input_json_delta"}, {"partial_json", "{\"id\":"}}) +
           BlockDelta(2, {{"type", "input_json_delta"}, {"partial_json", "1}"}}) +
           BlockStop(2) +
           Stop("tool_use", {{"output_tokens", 42}, {"output_tokens_details", {{"thinking_tokens", 30}}}});
}

Json ToolBlocks() {
    return Json::array({
        {{"type", "thinking"}, {"thinking", "Look at the graph."}, {"signature", "sig-one"}},
        {{"type", "text"}, {"text", "Checking now"}},
        {{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "inspect"}, {"input", {{"id", 1}}}},
    });
}

struct MessagesServer {
    AgentTest::ProviderServer provider;
    std::vector<Json> requests;
    std::vector<httplib::Headers> headers;
    std::function<void(const Json&, httplib::Response&)> reply;

    MessagesServer() {
        provider.server.Post("/v1/messages", [this](const auto& request, auto& response) {
            requests.push_back(Json::parse(request.body));
            headers.push_back(request.headers);
            reply(requests.back(), response);
        });
        provider.start();
    }

    Agent::Provider::Config config(Agent::Provider::Kind kind = Agent::Provider::Kind::Anthropic) const {
        auto config = provider.config(kind);
        config.api = Agent::Provider::Api::Messages;
        config.path = "/v1/messages";
        return config;
    }
};

void Respond(httplib::Response& response, const std::string& stream) {
    response.set_content(stream, "text/event-stream");
}

void Reject(httplib::Response& response, int status, const char* type, const char* message) {
    const Json body = {{"type", "error"}, {"error", {{"type", type}, {"message", message}}}};
    response.status = status;
    response.set_content(body.dump(), "application/json");
}

}  // namespace

TEST_CASE("Messages requests authenticate with Anthropic headers", "[agent][providers][messages][headers]") {
    const auto kind = GENERATE(Agent::Provider::Kind::Anthropic, Agent::Provider::Kind::Custom);

    MessagesServer server;
    server.reply = [](const Json&, auto& response) {
        Respond(response, TextReply("Hello"));
    };

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(server.config(kind)).stream(AgentTest::Request({{"user", "Hi"}}), {}, response) ==
            Result::SUCCESS);

    REQUIRE(server.headers.size() == 1);
    const auto& headers = server.headers.front();
    const auto value = [&](const char* name) {
        const auto found = headers.find(name);
        return found == headers.end() ? std::string{} : found->second;
    };
    CHECK(headers.find("Authorization") == headers.end());
    CHECK(value("x-api-key") == AgentTest::ProviderServer::keyValue);
    CHECK(value("anthropic-version") == "2026-10-21");
    CHECK(response.text == "Hello");
}

TEST_CASE("Messages encoder folds system prompts, maps tools, and merges roles",
          "[agent][providers][messages][encode]") {
    MessagesServer server;
    server.reply = [](const Json&, auto& response) {
        Respond(response, TextReply("Done"));
    };

    const std::vector<Agent::Provider::Function> tools{
        {"inspect", "Inspect a graph", R"({"type":"object","properties":{"id":{"type":"integer"}}})"},
    };
    const std::vector<Message> history{
        {"system", "Be brief."},
        {"system", "Use tools."},
        {"user", "Look", {}, "", {"data:image/png;base64,QUJD", "https://example.com/a.png"}},
        {"assistant", "Checking", {{"call_1", "inspect", R"({"id":1})"}, {"call_2", "inspect", "not json"}}},
        {"tool", "Missing node", {}, "call_1", {}, nullptr, true},
        {"tool", "{}", {}, "call_2"},
        {"user", "Thanks"},
        {"assistant", ""},
    };

    auto request = AgentTest::Request(history, tools);
    request.allowTools = false;

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(server.config()).stream(request, {}, response) == Result::SUCCESS);
    REQUIRE(server.requests.size() == 1);
    const auto& body = server.requests.front();

    CHECK(body["stream"] == true);
    CHECK(body["max_tokens"] == 32000);
    CHECK(body["system"] == Json::array({{
        {"type", "text"},
        {"text", "Be brief.\n\nUse tools."},
        {"cache_control", {{"type", "ephemeral"}}},
    }}));

    REQUIRE(body["tools"].size() == 1);
    CHECK(body["tools"][0]["name"] == "inspect");
    CHECK(body["tools"][0]["description"] == "Inspect a graph");
    CHECK(body["tools"][0]["input_schema"]["properties"]["id"]["type"] == "integer");
    CHECK(body["tool_choice"] == Json{{"type", "none"}});

    const auto& messages = body["messages"];
    REQUIRE(messages.size() == 3);
    CHECK(messages[0]["role"] == "user");
    CHECK(messages[1]["role"] == "assistant");
    CHECK(messages[2]["role"] == "user");

    CHECK(messages[0]["content"] == Json::array({
        {{"type", "text"}, {"text", "Look"}},
        {{"type", "image"}, {"source", {{"type", "base64"}, {"media_type", "image/png"}, {"data", "QUJD"}}}},
        {{"type", "image"}, {"source", {{"type", "url"}, {"url", "https://example.com/a.png"}}}},
    }));
    CHECK(messages[1]["content"] == Json::array({
        {{"type", "text"}, {"text", "Checking"}},
        {{"type", "tool_use"}, {"id", "call_1"}, {"name", "inspect"}, {"input", {{"id", 1}}}},
        {{"type", "tool_use"}, {"id", "call_2"}, {"name", "inspect"}, {"input", Json::object()}},
    }));
    CHECK(messages[2]["content"] == Json::array({
        {{"type", "tool_result"}, {"tool_use_id", "call_1"}, {"is_error", true},
         {"content", Json::array({{{"type", "text"}, {"text", "Missing node"}}})}},
        {{"type", "tool_result"}, {"tool_use_id", "call_2"},
         {"content", Json::array({{{"type", "text"}, {"text", "{}"}}})}},
        {{"type", "text"}, {"text", "Thanks"}, {"cache_control", {{"type", "ephemeral"}}}},
    }));
}

TEST_CASE("Tool image attachments fold into their own tool results", "[agent][providers][messages][encode]") {
    MessagesServer server;
    server.reply = [](const Json&, auto& response) {
        Respond(response, TextReply("Done"));
    };

    const Agent::Provider::FunctionCall capture{"call_1", "capture", "{}"};
    const Agent::Provider::FunctionCall inspect{"call_2", "inspect", "{}"};
    const Agent::Provider::FunctionCall view{"call_3", "view", "{}"};
    const std::string omitted = " [Older image omitted from context; capture the view again if needed.]";
    const std::vector<Message> history{
        {"user", "Capture"},
        {"assistant", "", {capture, inspect, view}},
        {"tool", "Captured", {}, "call_1"},
        {"tool", "Inspected", {}, "call_2"},
        {"tool", "Viewed", {}, "call_3"},
        {"user", capture.caption() + omitted},
        {"user", view.caption(), {}, "", {"data:image/png;base64,QUJD"}},
    };

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(server.config()).stream(AgentTest::Request(history), {}, response) == Result::SUCCESS);

    const auto& messages = server.requests.front()["messages"];
    REQUIRE(messages.size() == 3);
    const auto& results = messages[2]["content"];
    REQUIRE(results.size() == 3);
    CHECK(results[0]["content"] == Json::array({
        {{"type", "text"}, {"text", "Captured"}},
        {{"type", "text"}, {"text", capture.caption() + omitted}},
    }));
    CHECK(results[1]["content"] == Json::array({{{"type", "text"}, {"text", "Inspected"}}}));
    CHECK(results[2]["content"] == Json::array({
        {{"type", "text"}, {"text", "Viewed"}},
        {{"type", "text"}, {"text", view.caption()}},
        {{"type", "image"}, {"source", {{"type", "base64"}, {"media_type", "image/png"}, {"data", "QUJD"}}}},
    }));
    CHECK(results[2]["cache_control"] == Json{{"type", "ephemeral"}});
}

TEST_CASE("Messages encoder maps reasoning effort to adaptive thinking", "[agent][providers][messages][encode]") {
    const auto [effort, expected] = GENERATE(table<std::string, std::string>({
        {"minimal", "low"},
        {"low", "low"},
        {"medium", "medium"},
        {"high", "high"},
        {"xhigh", "xhigh"},
        {"max", "max"},
        {"custom", ""},
    }));
    const bool display = GENERATE(false, true);
    CAPTURE(effort, display);

    MessagesServer server;
    server.reply = [](const Json&, auto& response) {
        Respond(response, TextReply("Done"));
    };

    const std::vector<Message> history{{"user", "Hi"}};
    auto request = AgentTest::Request(history);
    request.reasoningEffort = effort;
    request.maxOutputTokens = 512;

    Agent::Provider::Callbacks callbacks;
    if (display) {
        callbacks.onReasoning = [](const auto&, bool) {};
    }

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(server.config()).stream(request, callbacks, response) == Result::SUCCESS);
    const auto& body = server.requests.front();

    CHECK(body["max_tokens"] == 512);
    CHECK(body["thinking"]["type"] == "adaptive");
    CHECK(body["thinking"].contains("display") == display);
    if (display) {
        CHECK(body["thinking"]["display"] == "summarized");
    }
    if (expected.empty()) {
        CHECK_FALSE(body.contains("output_config"));
    } else {
        CHECK(body["output_config"] == Json{{"effort", expected}});
    }
}

TEST_CASE("Messages encoder disables or omits thinking without a positive effort",
          "[agent][providers][messages][encode]") {
    const std::string effort = GENERATE("", "none");

    MessagesServer server;
    server.reply = [](const Json&, auto& response) {
        Respond(response, TextReply("Done"));
    };

    const std::vector<Message> history{{"user", "Hi"}};
    auto request = AgentTest::Request(history);
    request.reasoningEffort = effort;

    Agent::Provider::Response response;
    const Agent::Provider::Callbacks callbacks{.onReasoning = [](const auto&, bool) {}};
    REQUIRE(Agent::Provider(server.config()).stream(request, callbacks, response) == Result::SUCCESS);
    const auto& body = server.requests.front();

    CHECK_FALSE(body.contains("output_config"));
    if (effort.empty()) {
        CHECK_FALSE(body.contains("thinking"));
    } else {
        CHECK(body["thinking"] == Json{{"type", "disabled"}});
    }
}

TEST_CASE("Messages decoder streams text, thinking, tool input, and usage", "[agent][providers][messages][decode]") {
    MessagesServer server;
    server.reply = [](const Json&, auto& response) {
        Respond(response, ToolReply());
    };

    std::string streamed;
    std::string reasoning;
    std::vector<std::string> announced;
    const Agent::Provider::Callbacks callbacks{
        .onToken = [&](const std::string& token) {
            streamed += token;
        },
        .onReasoning = [&](const std::string& thought, bool summary) {
            CHECK(summary);
            reasoning += thought;
        },
        .onToolCall = [&](const std::string& name) {
            announced.push_back(name);
        },
    };

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(server.config()).stream(AgentTest::Request({{"user", "Inspect"}}), callbacks,
                                                    response) == Result::SUCCESS);

    CHECK(response.text == "Checking now");
    CHECK(streamed == response.text);
    CHECK(reasoning == "Look at the graph.");
    CHECK(announced == std::vector<std::string>{"inspect"});

    REQUIRE(response.toolCalls.size() == 1);
    CHECK(response.toolCalls[0].id == "toolu_1");
    CHECK(response.toolCalls[0].name == "inspect");
    CHECK(Json::parse(response.toolCalls[0].arguments) == Json{{"id", 1}});

    CHECK(response.usage.inputTokens == 95);
    CHECK(response.usage.cachedTokens == 80);
    CHECK(response.usage.cacheWriteTokens == 5);
    CHECK(response.usage.outputTokens == 42);
    CHECK(response.usage.reasoningTokens == 30);

    REQUIRE(response.state);
    CHECK(Json::parse(response.state->payload) == ToolBlocks());
}

TEST_CASE("Whitespace-only text blocks are not replayed", "[agent][providers][messages][replay]") {
    MessagesServer server;
    server.reply = [](const Json&, auto& response) {
        Respond(response, Start() +
                          BlockStart(0, {{"type", "text"}, {"text", ""}}) +
                          BlockDelta(0, {{"type", "text_delta"}, {"text", "\n\n"}}) +
                          BlockStop(0) +
                          BlockStart(1, {{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "inspect"},
                                         {"input", Json::object()}}) +
                          BlockStop(1) +
                          Stop("tool_use"));
    };

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(server.config()).stream(AgentTest::Request({{"user", "Inspect"}}), {}, response) ==
            Result::SUCCESS);
    REQUIRE(response.state);
    CHECK(Json::parse(response.state->payload) == Json::array({
        {{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "inspect"}, {"input", Json::object()}},
    }));
}

TEST_CASE("Messages stop reasons and stream errors become provider errors", "[agent][providers][messages][decode]") {
    const auto [stream, kind] = GENERATE(table<std::string, Kind>({
        {Start() + BlockStart(0, {{"type", "text"}, {"text", "Partial"}}) + BlockStop(0) + Stop("max_tokens"),
         Kind::OutputLimit},
        {Start() + Stop("model_context_window_exceeded"), Kind::OutputLimit},
        {Start() + BlockStart(0, {{"type", "tool_use"}, {"id", "t"}, {"name", "inspect"}, {"input", Json::object()}}) +
         BlockDelta(0, {{"type", "input_json_delta"}, {"partial_json", "{\"id\""}}) + BlockStop(0) + Stop("max_tokens"),
         Kind::OutputLimit},
        {Start() + Stop("refusal"), Kind::InvalidRequest},
        {Start() + Event({{"type", "error"}, {"error", {{"type", "overloaded_error"}, {"message", "Overloaded"}}}}),
         Kind::Server},
        {Start() + Event({{"type", "error"}, {"error", {{"type", "rate_limit_error"}, {"message", "Slow"}}}}),
         Kind::RateLimit},
        {Start() + BlockStart(0, {{"type", "text"}, {"text", "Cut"}}), Kind::Protocol},
        {Start() + BlockStart(0, {{"type", "tool_use"}, {"id", "t"}, {"name", "inspect"}, {"input", Json::object()}}) +
         BlockDelta(0, {{"type", "input_json_delta"}, {"partial_json", "{\"id\""}}) + BlockStop(0) + Stop("tool_use"),
         Kind::Protocol},
    }));

    MessagesServer server;
    server.reply = [&](const Json&, auto& response) {
        Respond(response, stream);
    };

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(server.config()).stream(AgentTest::Request({{"user", "Hi"}}), {}, response) ==
            Result::ERROR);
    CHECK(response.error.kind == kind);
    CHECK(server.requests.size() == 1);
}

TEST_CASE("Anthropic HTTP error types map onto provider error kinds", "[agent][providers][messages][errors]") {
    const auto [status, type, kind] = GENERATE(table<int, std::string, Kind>({
        {401, "authentication_error", Kind::Authentication},
        {403, "permission_error", Kind::Permission},
        {404, "not_found_error", Kind::NotFound},
        {429, "rate_limit_error", Kind::RateLimit},
        {402, "billing_error", Kind::Quota},
        {400, "billing_error", Kind::Quota},
        {529, "overloaded_error", Kind::Server},
        {500, "timeout_error", Kind::Timeout},
        {413, "request_too_large", Kind::ContextLength},
        {400, "request_too_large", Kind::ContextLength},
    }));
    CAPTURE(status, type);

    MessagesServer server;
    server.reply = [&](const Json&, auto& response) {
        Reject(response, status, type.c_str(), "Request failed");
    };

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(server.config()).stream(AgentTest::Request({{"user", "Hi"}}), {}, response) ==
            Result::ERROR);
    CHECK(response.error.kind == kind);
    CHECK(response.error.status == status);
}

TEST_CASE("Messages replay returns stored content blocks unchanged", "[agent][providers][messages][replay]") {
    const bool sameModel = GENERATE(true, false);

    MessagesServer server;
    server.reply = [&](const Json&, auto& response) {
        Respond(response, server.requests.size() == 1 ? ToolReply() : TextReply("Done"));
    };

    const Agent::Provider client(server.config());
    const Agent::Provider::Callbacks callbacks{.onReasoning = [](const auto&, bool) {}};

    Agent::Provider::Response first;
    REQUIRE(client.stream(AgentTest::Request({{"user", "Inspect"}}), callbacks, first) == Result::SUCCESS);
    REQUIRE(first.state);

    const std::vector<Message> history{
        {"user", "Inspect"},
        {"assistant", first.text, first.toolCalls, "", {}, first.state},
        {"tool", "{\"ok\":true}", {}, "toolu_1"},
    };
    auto request = AgentTest::Request(history);
    if (!sameModel) {
        request.model = "other-model";
    }

    Agent::Provider::Response second;
    REQUIRE(client.stream(request, callbacks, second) == Result::SUCCESS);
    REQUIRE(server.requests.size() == 2);

    const auto& messages = server.requests[1]["messages"];
    REQUIRE(messages.size() == 3);
    CHECK(messages[1]["role"] == "assistant");
    if (sameModel) {
        CHECK(messages[1]["content"] == ToolBlocks());
    } else {
        CHECK(messages[1]["content"] == Json::array({
            {{"type", "text"}, {"text", "Checking now"}},
            {{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "inspect"}, {"input", {{"id", 1}}}},
        }));
    }
    CHECK(messages[2]["content"][0]["type"] == "tool_result");
    CHECK(messages[2]["content"][0]["tool_use_id"] == "toolu_1");
    CHECK(messages[2]["content"][0]["cache_control"] == Json{{"type", "ephemeral"}});
}

TEST_CASE("Rejected thinking options are dropped and retried once", "[agent][providers][messages][retry]") {
    const auto [message, relevant] = GENERATE(table<std::string, bool>({
        {"thinking.type: adaptive thinking is not supported on this model.", true},
        {"output_config: Extra inputs are not permitted", true},
        {"effort: Input should be 'low', 'medium' or 'high'", true},
        {"thinking.type: 'disabled' is invalid for this model. Use 'between_tools'.", true},
        {"messages: roles must alternate between user and assistant", false},
    }));
    CAPTURE(message);

    MessagesServer server;
    server.reply = [&](const Json& body, auto& response) {
        if (body.contains("thinking") || body.contains("output_config") || !relevant) {
            Reject(response, 400, "invalid_request_error", message.c_str());
            return;
        }
        Respond(response, TextReply("Done"));
    };

    auto config = server.config();
    config.retries = 3;
    const Agent::Provider client(config);

    const std::vector<Message> history{{"system", "Be brief."}, {"user", "Hi"}};
    auto request = AgentTest::Request(history);
    request.reasoningEffort = "high";

    Agent::Provider::Response response;
    const auto result = client.stream(request, {}, response);
    REQUIRE(result == (relevant ? Result::SUCCESS : Result::ERROR));
    REQUIRE(server.requests.size() == (relevant ? 2u : 1u));
    CHECK(server.requests[0]["output_config"] == Json{{"effort", "high"}});
    if (!relevant) {
        CHECK(response.error.status == 400);
        return;
    }

    CHECK_FALSE(server.requests[1].contains("thinking"));
    CHECK_FALSE(server.requests[1].contains("output_config"));
    CHECK(server.requests[1]["messages"] == server.requests[0]["messages"]);
    CHECK(server.requests[1]["system"] == server.requests[0]["system"]);

    REQUIRE(client.stream(request, {}, response) == Result::SUCCESS);
    REQUIRE(server.requests.size() == 3);
    CHECK(server.requests[2] == server.requests[1]);
}

TEST_CASE("Thinking bound to a different conversation is dropped and retried once",
          "[agent][providers][messages][retry]") {
    const bool persistent = GENERATE(false, true);

    MessagesServer server;
    server.reply = [&](const Json& body, auto& response) {
        if (server.requests.size() == 1) {
            Respond(response, ToolReply());
            return;
        }

        bool thinking = false;
        for (const auto& message : body["messages"]) {
            for (const auto& block : message["content"]) {
                thinking |= block.value("type", "") == "thinking";
            }
        }
        if (thinking || persistent) {
            Reject(response, 400, "invalid_request_error",
                   "messages.1.content.0: Invalid `signature` in `thinking` block. "
                   "The block is bound to a different conversation.");
            return;
        }
        Respond(response, TextReply("Done"));
    };

    auto config = server.config();
    config.retries = 3;
    const Agent::Provider client(config);

    Agent::Provider::Response first;
    REQUIRE(client.stream(AgentTest::Request({{"user", "Inspect"}}), {}, first) == Result::SUCCESS);

    const std::vector<Message> history{
        {"user", "Inspect [Older image omitted from context; capture the view again if needed.]"},
        {"assistant", first.text, first.toolCalls, "", {}, first.state},
        {"tool", "{}", {}, "toolu_1"},
    };
    const auto request = AgentTest::Request(history);

    Agent::Provider::Response second;
    REQUIRE(client.stream(request, {}, second) == (persistent ? Result::ERROR : Result::SUCCESS));
    REQUIRE(server.requests.size() == 3);
    CHECK(server.requests[1]["messages"][1]["content"] == ToolBlocks());
    CHECK(server.requests[2]["messages"][1]["content"] == Json::array({ToolBlocks()[1], ToolBlocks()[2]}));

    REQUIRE(client.stream(request, {}, second) == (persistent ? Result::ERROR : Result::SUCCESS));
    REQUIRE(server.requests.size() == 5);
    CHECK(server.requests[3]["messages"][1]["content"] == ToolBlocks());
}

TEST_CASE("Thinking phrases do not drop options from other request formats", "[agent][providers][messages][retry]") {
    AgentTest::ProviderServer provider;
    int count = 0;
    provider.server.Post("/chat", [&](const auto&, auto& response) {
        ++count;
        response.status = 400;
        response.set_content(R"({"error":{"message":"thinking is not supported"}})", "application/json");
    });
    provider.start();

    auto config = provider.config();
    config.retries = 3;

    const std::vector<Message> history{{"user", "Hi"}};
    auto request = AgentTest::Request(history);
    request.reasoningEffort = "high";

    Agent::Provider::Response response;
    const Agent::Provider::Callbacks callbacks{.onReasoning = [](const auto&, bool) {}};
    REQUIRE(Agent::Provider(config).stream(request, callbacks, response) == Result::ERROR);
    CHECK(count == 1);
}

#endif
