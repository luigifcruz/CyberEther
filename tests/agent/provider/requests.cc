#include "jetstream/config.hh"

#if defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include "agent_provider.hh"

#include <catch2/generators/catch_generators.hpp>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <future>

using namespace Jetstream;
using namespace std::chrono_literals;

namespace {

using Kind = Agent::Provider::Error::Kind;

using AgentTest::Sse::Delta;
using AgentTest::Sse::Done;
using AgentTest::Sse::Event;

Result Chat(const Agent::Provider& client, Agent::Provider::Error& error, std::string* text = nullptr) {
    Agent::Provider::Response response;
    const auto result = client.stream(AgentTest::Request({{"user", "Hello"}}), {}, response);

    error = response.error;
    if (text) {
        *text = response.text;
    }
    return result;
}

}  // namespace

TEST_CASE("Request budgets preserve schemas and detailed usage survives both codecs", "[agent][providers][usage]") {
    const bool responses = GENERATE(false, true);
    const auto path = responses ? "/v1/responses" : "/chat";

    AgentTest::ProviderServer provider;
    nlohmann::json captured;
    provider.server.Post(path, [&](const auto& input, auto& output) {
        captured = nlohmann::json::parse(input.body);

        const char* inputKey = responses ? "input_tokens" : "prompt_tokens";
        const char* outputKey = responses ? "output_tokens" : "completion_tokens";
        const char* inputDetailsKey = responses ? "input_tokens_details" : "prompt_tokens_details";
        const char* outputDetailsKey = responses ? "output_tokens_details" : "completion_tokens_details";
        nlohmann::json usage = {
            {inputKey, 100},
            {outputKey, 20},
            {inputDetailsKey, {{"cached_tokens", 80}, {"cache_write_tokens", 10}}},
            {outputDetailsKey, {{"reasoning_tokens", 15}}},
            {"cost", 0.01},
        };

        std::string stream;
        if (responses) {
            stream = Event({{"type", "response.completed"}, {"response", {{"usage", usage}}}});
        } else {
            const nlohmann::json event = {{"choices", nlohmann::json::array()}, {"usage", usage}};
            stream = Delta({{"content", "Done"}}) + Event(event) + Done();
        }
        output.set_content(stream, "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    config.path = path;
    config.api = responses ? Agent::Provider::Api::Responses : Agent::Provider::Api::ChatCompletions;
    Agent::Provider client(config);

    const std::vector<Agent::Provider::Message> messages{{"user", "Summarize"}};
    const std::vector<Agent::Provider::Function> tools{{"inspect", "Inspect", R"({"type":"object"})"}};
    auto request = AgentTest::Request(messages, tools);
    request.allowTools = false;
    request.maxOutputTokens = 512;
    request.reasoningEffort = "high";
    Agent::Provider::Response response;
    REQUIRE(client.stream(request, {}, response) == Result::SUCCESS);

    CHECK(captured["tools"].size() == 1);
    CHECK(captured["tool_choice"] == "none");
    CHECK(captured[responses ? "max_output_tokens" : "max_tokens"] == 512);
    CHECK(response.usage.cachedTokens == 80);
    CHECK(response.usage.cacheWriteTokens == 10);
    CHECK(response.usage.reasoningTokens == 15);
    REQUIRE(response.usage.cost);
    CHECK(*response.usage.cost == 0.01);
}

TEST_CASE("Model discovery retains provider output allowances", "[agent][providers][models]") {
    AgentTest::ProviderServer provider;
    provider.server.Get("/models", [](const auto&, auto& response) {
        response.set_content(R"({"data":[
            {"id":"direct","context_length":32768,"max_output_tokens":8192},
            {"id":"routed","context_length":1000000,"top_provider":{"max_completion_tokens":384000}}
        ]})", "application/json");
    });
    provider.start();

    Agent::Provider client(provider.config());
    std::vector<Agent::Provider::Model> models;
    REQUIRE(client.models(models) == Result::SUCCESS);

    REQUIRE(models.size() == 2);
    CHECK(models[0].outputLength == 8192);
    CHECK(models[1].outputLength == 384000);
}

TEST_CASE("Provider headers are limited to the selected integration", "[agent][providers][headers]") {
    const auto kind = GENERATE(Agent::Provider::Kind::OpenAI,
                               Agent::Provider::Kind::OpenRouter,
                               Agent::Provider::Kind::OpenCodeGo,
                               Agent::Provider::Kind::Fireworks,
                               Agent::Provider::Kind::Custom);

    AgentTest::ProviderServer provider;
    std::promise<httplib::Headers> captured;
    auto headersFuture = captured.get_future();
    provider.server.Post("/chat", [&](const auto& request, auto& response) {
        captured.set_value(request.headers);
        response.set_content(Delta({{"content", "Hello"}}) + Done(), "text/event-stream");
    });
    provider.start();

    const Agent::Provider client(provider.config(kind));
    Agent::Provider::Error error;
    REQUIRE(Chat(client, error) == Result::SUCCESS);

    const auto headers = headersFuture.get();
    const auto contains = [&](const char* name) {
        return headers.find(name) != headers.end();
    };
    const bool openRouter = kind == Agent::Provider::Kind::OpenRouter;
    const bool openCodeGo = kind == Agent::Provider::Kind::OpenCodeGo;

    REQUIRE(contains("Authorization"));
    CHECK(contains("HTTP-Referer") == openRouter);
    CHECK(contains("X-Title") == openRouter);
    CHECK(contains("x-session-id") == openRouter);
    CHECK(contains("x-opencode-session") == openCodeGo);
}

TEST_CASE("OpenAI Responses streams function calls and accepts tool results on the next turn",
          "[agent][providers][responses]") {
    AgentTest::ProviderServer provider;
    std::vector<nlohmann::json> requests;
    provider.server.Post("/v1/responses", [&](const auto& request, auto& response) {
        const auto body = nlohmann::json::parse(request.body);
        requests.push_back(body);

        if (requests.size() == 1) {
            response.set_content(
                "event: response.output_text.delta\n"
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Checking\"}\n\n"
                "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\","
                "\"call_id\":\"call_1\",\"name\":\"inspect\",\"arguments\":\"{\\\"id\\\":1}\"}}\n\n"
                "data: {\"type\":\"response.completed\"}\n\ndata: [DONE]\n\n",
                "text/event-stream");
            return;
        }

        response.set_content(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Done\"}\n\n"
            "data: {\"type\":\"response.completed\"}\n\ndata: [DONE]\n\n",
            "text/event-stream");
    });
    provider.start();

    auto config = provider.config(Agent::Provider::Kind::OpenAI);
    config.path = "/v1/responses";
    config.api = Agent::Provider::Api::Responses;
    const Agent::Provider client(config);

    const std::vector<Agent::Provider::Function> tools{
        {"inspect", "Inspect a graph", R"({"type":"object","properties":{"id":{"type":"integer"}}})"},
    };
    const auto ask = [&](const std::vector<Agent::Provider::Message>& messages) {
        auto request = AgentTest::Request(messages, tools);
        request.model = "gpt-6-luna";
        return request;
    };

    std::string streamed;
    const Agent::Provider::Callbacks callbacks{.onToken = [&](const std::string& token) {
        streamed += token;
    }};

    Agent::Provider::Response response;
    REQUIRE(client.stream(ask({{"user", "Inspect this"}}), callbacks, response) ==
            Result::SUCCESS);

    auto calls = response.toolCalls;
    auto text = response.text;
    CHECK(text == "Checking");
    CHECK(streamed == text);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].id == "call_1");
    CHECK(calls[0].name == "inspect");
    CHECK(calls[0].arguments == R"({"id":1})");

    const std::vector<Agent::Provider::Message> history = {
        {"user", "Inspect this"},
        {"assistant", text, calls, ""},
        {"tool", R"({"found":true})", {}, "call_1"},
        {"user", "Screenshot", {}, "", {"data:image/png;base64,AA=="}},
    };
    REQUIRE(client.stream(ask(history), {}, response) == Result::SUCCESS);

    CHECK(response.text == "Done");
    CHECK(response.toolCalls.empty());

    REQUIRE(requests.size() == 2);
    CHECK(requests[0]["model"] == "gpt-6-luna");
    CHECK(requests[0]["store"] == false);
    CHECK(requests[0].contains("input"));
    CHECK_FALSE(requests[0].contains("messages"));
    CHECK_FALSE(requests[0].contains("reasoning"));
    CHECK_FALSE(requests[0].contains("reasoning_effort"));
    CHECK(requests[0]["tools"][0]["type"] == "function");
    CHECK(requests[0]["tools"][0]["strict"] == false);
    CHECK(requests[1]["input"][1]["role"] == "assistant");
    CHECK(requests[1]["input"][2]["type"] == "function_call");
    CHECK(requests[1]["input"][2]["call_id"] == "call_1");
    CHECK(requests[1]["input"][3]["type"] == "function_call_output");
    CHECK(requests[1]["input"][3]["output"] == R"({"found":true})");
    CHECK(requests[1]["input"][4]["content"][0]["type"] == "input_text");
    CHECK(requests[1]["input"][4]["content"][1]["type"] == "input_image");
    CHECK(requests[1]["input"][4]["content"][1]["image_url"] == "data:image/png;base64,AA==");
}

TEST_CASE("OpenAI Responses reports failed and truncated streams", "[agent][providers][responses]") {
    const bool failed = GENERATE(false, true);

    AgentTest::ProviderServer provider;
    provider.server.Post("/v1/responses", [failed](const auto&, auto& response) {
        const char* stream = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Partial\"}\n\ndata: [DONE]\n\n";
        if (failed) {
            stream = "data: {\"type\":\"response.failed\",\"response\":{\"error\":"
                     "{\"message\":\"Tool execution rejected\",\"code\":\"bad_tool\"}}}\n\n";
        }
        response.set_content(stream, "text/event-stream");
    });
    provider.start();

    auto config = provider.config(Agent::Provider::Kind::OpenAI);
    config.path = "/v1/responses";
    config.api = Agent::Provider::Api::Responses;

    Agent::Provider::Error error;
    REQUIRE(Chat(Agent::Provider(config), error) == Result::ERROR);

    const char* expected = failed ? "Tool execution rejected" : "no completion data";
    CHECK(error.message.find(expected) != std::string::npos);
    if (failed) {
        CHECK(error.code == "bad_tool");
    }
}

TEST_CASE("HTTP diagnostics preserve status, provider detail, and retry information", "[agent][providers][errors]") {
    const int status = GENERATE(400, 401, 402, 403, 404, 408, 429, 500, 503, 504);
    CAPTURE(status);

    AgentTest::ProviderServer provider;
    const auto handler = [status](const auto&, auto& response) {
        response.status = status;
        response.set_header("Retry-After", "17");
        response.set_content(R"({"error":{"message":"Provider diagnostic","code":"test_code"}})", "application/json");
    };
    provider.server.Post("/chat", handler);
    provider.server.Get("/models", handler);
    provider.start();

    const Agent::Provider client(provider.config());
    Agent::Provider::Error error;

    const auto verify = [&] {
        CHECK(error.status == status);
        CHECK(error.code == "test_code");
        CHECK(error.retryAfter == "17");
        CHECK(error.message.find("Provider diagnostic") != std::string::npos);
        CHECK(error.message.find("HTTP " + std::to_string(status)) != std::string::npos);
        CHECK(error.message.find_first_of("\r\n\t") == std::string::npos);
        CHECK(error.message.find("Provider code:") == std::string::npos);

        if (status != 401) {
            CHECK(error.kind != Kind::Authentication);
            CHECK(error.message.find("API key") == std::string::npos);
        }
        if (status == 429) {
            CHECK(error.kind == Kind::RateLimit);
        }
        if (status == 402) {
            CHECK(error.kind == Kind::Quota);
        }
        if (status == 403) {
            CHECK(error.kind == Kind::Permission);
        }
        if (status == 404) {
            CHECK(error.kind == Kind::NotFound);
        }
        if (status == 408 || status == 504) {
            CHECK(error.kind == Kind::Timeout);
        }
    };

    REQUIRE(Chat(client, error) == Result::ERROR);
    verify();

    std::vector<Agent::Provider::Model> models{{"stale", "Stale"}};
    REQUIRE(client.models(models, &error) == Result::ERROR);
    CHECK(models.empty());
    verify();
}

TEST_CASE("Service errors show one useful explanation and retain diagnostic metadata",
          "[agent][providers][errors]") {
    const bool hasDetail = GENERATE(false, true);

    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [hasDetail](const auto&, auto& response) {
        const char* message = hasDetail ? "  Upstream request failed:\r\n\tEndpoint is unavailable.  " : "";
        const nlohmann::json body = {{"error", {
            {"message", message},
            {"code", "server_error"},
        }}};

        response.status = 503;
        response.set_content(body.dump(), "application/json");
    });
    provider.start();

    const auto config = provider.config(Agent::Provider::Kind::OpenCodeGo);
    Agent::Provider::Error error;
    REQUIRE(Chat(Agent::Provider(config), error) == Result::ERROR);

    std::string expected = "OpenCode Go: Service unavailable (HTTP 503). Try again later.";
    if (hasDetail) {
        expected = "OpenCode Go: Endpoint is unavailable (HTTP 503).";
    }
    CHECK(error.message == expected);
    CHECK(error.kind == Kind::Server);
    CHECK(error.status == 503);
    CHECK(error.code == "server_error");
}

TEST_CASE("Streaming provider errors survive HTTP 200 and partial output", "[agent][providers][errors]") {
    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [](const auto&, auto& response) {
        const std::string stream = Delta({{"content", "Partial answer"}}) +
                                   "event: error\ndata: {\"error\":{\"code\":429,"
                                   "\"message\":\"Upstream capacity exhausted\"}}\n\n";
        response.set_content(stream, "text/event-stream");
    });
    provider.start();

    Agent::Provider::Error error;
    std::string text;
    REQUIRE(Chat(Agent::Provider(provider.config()), error, &text) == Result::ERROR);

    CHECK(error.kind == Kind::RateLimit);
    CHECK(error.status == 429);
    CHECK(error.message.find("Upstream capacity exhausted") != std::string::npos);
    CHECK(text == "Partial answer");
}

TEST_CASE("Malformed, empty, and truncated streams fail instead of succeeding silently",
          "[agent][providers][stream]") {
    const auto body = GENERATE(
        std::string("data: not-json\n\n"),
        std::string("data: {}\n\n"),
        std::string("data: [DONE]\n\n"),
        std::string("<html>not an API</html>"),
        Delta({{"content", "Incomplete"}}));
    CAPTURE(body);

    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [body](const auto&, auto& response) {
        response.set_content(body, "text/event-stream");
    });
    provider.start();

    Agent::Provider::Error error;
    REQUIRE(Chat(Agent::Provider(provider.config()), error) == Result::ERROR);

    CHECK(error.kind == Kind::Protocol);
    CHECK_FALSE(error.message.empty());
    CHECK(error.message.find("API key") == std::string::npos);
}

TEST_CASE("SSE decoding handles CRLF, comments, multiline data, and fragmented bytes",
          "[agent][providers][stream]") {
    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [](const auto&, auto& response) {
        const std::string content =
            ": keepalive\r\n\r\nevent: message\r\ndata: {\"choices\":[\r\n"
            "data: {\"delta\":{\"content\":\"Hello\"}}]}\r\n\r\n"
            "data: [DONE]";

        const auto provide = [content](size_t offset, httplib::DataSink& sink) {
            const auto length = std::min<size_t>(3, content.size() - offset);
            if (!sink.write(content.data() + offset, length)) {
                return false;
            }
            if (offset + length == content.size()) {
                sink.done();
            }
            return true;
        };
        response.set_chunked_content_provider("text/event-stream", provide);
    });
    provider.start();

    Agent::Provider::Error error;
    std::string text;
    REQUIRE(Chat(Agent::Provider(provider.config()), error, &text) == Result::SUCCESS);

    CHECK(text == "Hello");
    CHECK_FALSE(error);
}

TEST_CASE("Missing credentials and unauthenticated local endpoints are distinct",
          "[agent][providers][authentication]") {
    AgentTest::ProviderServer provider;
    std::atomic<int> requests{0};
    provider.server.Post("/chat", [&](const auto&, auto& response) {
        requests++;
        response.set_content(Delta({{"content", "Local"}}) + Done(), "text/event-stream");
    });
    provider.start();

    auto config = provider.config(Agent::Provider::Kind::OpenAI);
    config.apiKey = {};

    Agent::Provider::Error error;
    REQUIRE(Chat(Agent::Provider(config), error) == Result::ERROR);
    CHECK(error.kind == Kind::MissingCredentials);
    CHECK(requests.load() == 0);

    config.kind = Agent::Provider::Kind::Custom;
    REQUIRE(Chat(Agent::Provider(config), error) == Result::SUCCESS);
    CHECK_FALSE(error);
    CHECK(requests.load() == 1);
}

TEST_CASE("Provider diagnostics redact echoed credentials and distinguish quota", "[agent][providers][errors]") {
    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [](const auto&, auto& response) {
        const nlohmann::json body = {{"error", {
            {"message", std::string("No credits for ") + AgentTest::ProviderServer::keyValue},
            {"code", "insufficient_quota"},
        }}};

        response.status = 429;
        response.set_content(body.dump(), "application/json");
    });
    provider.start();

    Agent::Provider::Error error;
    REQUIRE(Chat(Agent::Provider(provider.config()), error) == Result::ERROR);

    CHECK(error.kind == Kind::Quota);
    CHECK(error.message.find(provider.keyValue) == std::string::npos);
    CHECK(error.message.find("[redacted]") != std::string::npos);
}

TEST_CASE("Cancellation interrupts a provider that stops sending bytes", "[agent][providers][cancellation]") {
    AgentTest::ProviderServer provider;
    std::promise<void> entered;
    std::promise<void> release;
    auto received = entered.get_future();
    auto released = release.get_future().share();
    provider.server.Post("/chat", [&](const auto&, auto& response) {
        const auto provide = [&](size_t, httplib::DataSink& sink) {
            const auto chunk = Delta({{"content", "Starting"}});
            sink.write(chunk.data(), chunk.size());
            entered.set_value();
            released.wait();
            sink.done();
            return true;
        };
        response.set_chunked_content_provider("text/event-stream", provide);
    });
    provider.start();

    std::atomic<bool> abort{false};
    auto result = std::async(std::launch::async, [&] {
        const Agent::Provider client(provider.config());
        const Agent::Provider::Callbacks callbacks{.shouldAbort = [&] {
            return abort.load();
        }};

        Agent::Provider::Response response;
        const auto status = client.stream(AgentTest::Request({{"user", "Hello"}}), callbacks, response);
        return std::pair{status, response.error};
    });

    const bool started = received.wait_for(3s) == std::future_status::ready;
    abort.store(true);
    const bool stopped = result.wait_for(1s) == std::future_status::ready;
    release.set_value();
    provider.stop();

    REQUIRE(started);
    REQUIRE(stopped);

    const auto [status, error] = result.get();
    CHECK(status == Result::ERROR);
    CHECK(error.kind == Kind::Cancelled);
}

TEST_CASE("Cancellation interrupts in-flight model discovery", "[agent][providers][cancellation]") {
    AgentTest::ProviderServer provider;
    std::promise<void> entered;
    std::promise<void> release;
    auto received = entered.get_future();
    auto released = release.get_future().share();
    provider.server.Get("/models", [&](const auto&, auto& response) {
        const auto provide = [&](size_t, httplib::DataSink& sink) {
            const std::string chunk = R"({"data":[)";
            if (!sink.write(chunk.data(), chunk.size())) {
                return false;
            }
            entered.set_value();
            released.wait();
            sink.done();
            return true;
        };
        response.set_chunked_content_provider("application/json", provide);
    });
    provider.start();

    std::atomic<bool> abort{false};
    auto result = std::async(std::launch::async, [&] {
        const Agent::Provider client(provider.config());
        std::vector<Agent::Provider::Model> models;
        Agent::Provider::Error error;
        const auto status = client.models(models, &error, [&] {
            return abort.load();
        });
        return std::pair{status, error};
    });

    const bool started = received.wait_for(3s) == std::future_status::ready;
    abort.store(true);
    const bool stopped = result.wait_for(1s) == std::future_status::ready;
    release.set_value();
    provider.stop();

    REQUIRE(started);
    REQUIRE(stopped);

    const auto [status, error] = result.get();
    CHECK(status == Result::ERROR);
    CHECK(error.kind == Kind::Cancelled);
}

TEST_CASE("Model discovery reports malformed responses without blaming credentials",
          "[agent][providers][models]") {
    const auto body = GENERATE(std::string("not JSON"),
                               std::string("[]"),
                               std::string(R"({"data":"invalid"})"));

    AgentTest::ProviderServer provider;
    provider.server.Get("/models", [body](const auto&, auto& response) {
        response.set_content(body, "application/json");
    });
    provider.start();

    std::vector<Agent::Provider::Model> models;
    Agent::Provider::Error error;
    Agent::Provider client(provider.config());
    REQUIRE(client.models(models, &error) == Result::ERROR);

    CHECK(error.kind == Kind::Protocol);
    CHECK(error.message.find("API key") == std::string::npos);
}

TEST_CASE("Connection failures and request timeouts are different diagnostics", "[agent][providers][network]") {
    const bool timeout = GENERATE(false, true);

    AgentTest::ProviderServer provider;
    provider.server.Post("/chat", [](const auto&, auto& response) {
        std::this_thread::sleep_for(1500ms);
        response.set_content(Delta({{"content", "Late"}}) + Done(), "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    config.timeoutSeconds = timeout ? 1 : 5;
    if (!timeout) {
        provider.stop();
    }

    Agent::Provider::Error error;
    REQUIRE(Chat(Agent::Provider(config), error) == Result::ERROR);

    CHECK(error.kind == (timeout ? Kind::Timeout : Kind::Connection));
    CHECK(error.message.find("API key") == std::string::npos);
}

#endif
