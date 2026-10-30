#include "jetstream/config.hh"

#if defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include "agent_provider.hh"
#include "agent/provider/support/base.hh"
#include "jetstream/logger.hh"

#include <catch2/generators/catch_generators.hpp>
#include <nlohmann/json.hpp>

#include <atomic>
#include <future>
#include <limits>
#include <sstream>

using namespace Jetstream;
using namespace std::chrono_literals;
using Json = nlohmann::json;

namespace {

constexpr const char* completion = "data: {\"choices\":[{\"delta\":{\"content\":\"Done\"}}]}\n\ndata: [DONE]\n\n";
constexpr const char* responseCompleted = "data: {\"type\":\"response.completed\"}\n\n";

struct LogCapture {
    std::ostringstream output;
    int level = _JST_LOG_DEBUG_LEVEL();
    std::ostream* previous = &JST_LOG_SINK();

    LogCapture() {
        JST_LOG_SET_DEBUG_LEVEL(2);
        JST_LOG_SET_SINK(&output);
    }

    ~LogCapture() {
        JST_LOG_SET_SINK(previous);
        JST_LOG_SET_DEBUG_LEVEL(level);
    }
};

void UseResponses(Agent::Provider::Config& config) {
    config.path = "/v1/responses";
    config.api = Agent::Provider::Api::Responses;
}

}  // namespace

TEST_CASE("Retry policy honors seconds and all HTTP date forms without shortening delays",
          "[agent][providers][retry]") {
    using namespace std::chrono;

    const auto now = sys_days(year{2026}/10/1) + 12h + 125ms;
    Agent::Provider::Error error{Agent::Provider::Error::Kind::RateLimit, 429, "Busy"};

    const auto dates = {
        "Thu, 01 Oct 2026 12:00:05 GMT",
        "Thursday, 01-Oct-26 12:00:05 GMT",
        "Thu Oct  1 12:00:05 2026",
    };
    for (const auto* date : dates) {
        error.retryAfter = date;
        CAPTURE(date);
        CHECK(Agent::Detail::Retry::Delay(error, 0, 3, now) == 4875ms);
    }

    error.retryAfter = " 120\t";
    CHECK(Agent::Detail::Retry::Delay(error, 0, 3, now) == 120s);

    error.retryAfter = "0";
    CHECK(Agent::Detail::Retry::Delay(error, 0, 3, now) == 0ms);

    error.retryAfter = "Thu, 01 Oct 2026 11:59:59 GMT";
    CHECK(Agent::Detail::Retry::Delay(error, 0, 3, now) == 0ms);

    for (const auto* excessive : {"121", "99999999999999999999999999999", "Thu, 01 Oct 2026 13:00:00 GMT"}) {
        error.retryAfter = excessive;
        CAPTURE(excessive);
        CHECK_FALSE(Agent::Detail::Retry::Delay(error, 0, 3, now));
    }

    error.retryAfter = "invalid date";
    const auto first = Agent::Detail::Retry::Delay(error, 0, 3, now);
    REQUIRE(first);
    CHECK(*first >= 750ms);
    CHECK(*first <= 1000ms);

    const auto capped = Agent::Detail::Retry::Delay(error, 64, std::numeric_limits<U32>::max(), now);
    REQUIRE(capped);
    CHECK(*capped >= 22500ms);
    CHECK(*capped <= 30000ms);

    CHECK_FALSE(Agent::Detail::Retry::Delay(error, 3, 3, now));

    error.kind = Agent::Provider::Error::Kind::Authentication;
    CHECK_FALSE(Agent::Detail::Retry::Delay(error, 0, 3, now));
}

TEST_CASE("Retry and reasoning fallback warnings redact echoed credentials", "[agent][providers][retry]") {
    const int status = GENERATE(429, 400);

    AgentTest::ProviderServer provider;
    int count = 0;
    provider.server.Post("/chat", [&](const auto&, auto& response) {
        if (++count != 1) {
            response.set_content(completion, "text/event-stream");
            return;
        }

        const std::string prefix = status == 400 ? "Unsupported parameter: reasoning for " : "Rate limited key ";
        const Json body = {{"error", {
            {"message", prefix + provider.keyValue},
            {"code", provider.keyValue},
        }}};

        response.status = status;
        response.set_header("Retry-After", "0");
        response.set_content(body.dump(), "application/json");
    });
    provider.start();

    auto config = provider.config();
    config.retries = 1;

    LogCapture logs;
    Agent::Provider::Response response;
    const Agent::Provider::Callbacks callbacks{.onReasoning = [](const auto&, bool) {}};
    REQUIRE(Agent::Provider(config).stream(AgentTest::Request({{"user", "Hello"}}), callbacks, response) ==
            Result::SUCCESS);

    CHECK(count == 2);
    CHECK(logs.output.str().find(provider.keyValue) == std::string::npos);
    CHECK(logs.output.str().find("[redacted]") != std::string::npos);
}

TEST_CASE("Reasoning fallback removes request options but preserves replayed state",
          "[agent][providers][retry]") {
    const bool responses = GENERATE(false, true);
    const bool priorState = GENERATE(false, true);

    AgentTest::ProviderServer provider;
    std::vector<Json> requests;
    provider.server.Post(responses ? "/v1/responses" : "/chat", [&](const auto& request, auto& response) {
        requests.push_back(Json::parse(request.body));

        if (requests.back().contains("reasoning")) {
            response.status = 400;
            response.set_content(R"({"error":{"message":"Unsupported reasoning option","param":"reasoning"}})",
                                 "application/json");
            return;
        }

        response.set_content(responses ? responseCompleted : completion, "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    if (responses) {
        UseResponses(config);
    }

    Json state = {{"reasoning_content", "Earlier thought"}};
    if (responses) {
        state = Json::array({
            {{"type", "reasoning"}, {"id", "rs-one"}, {"summary", Json::array()}, {"encrypted_content", "opaque"}},
            {{"type", "message"}, {"role", "assistant"}, {"phase", "final_answer"}, {"content", "Before"}},
        });
    }

    std::vector<Agent::Provider::Message> history{{"user", "Hello"}};
    if (priorState) {
        const Agent::Provider::State saved{config.host, config.path, AgentTest::ProviderServer::modelValue, state.dump()};
        const auto shared = std::make_shared<const Agent::Provider::State>(saved);
        history.push_back({"assistant", "Before", {}, "", {}, shared});
    }

    const Agent::Provider client(config);
    const Agent::Provider::Callbacks callbacks{.onReasoning = [](const auto&, bool) {}};
    Agent::Provider::Response response;

    REQUIRE(client.stream(AgentTest::Request(history), callbacks, response) == Result::SUCCESS);
    REQUIRE(requests.size() == 2);
    CHECK_FALSE(requests.back().contains("reasoning"));
    CHECK(requests.back().contains("include") == responses);

    const auto* key = responses ? "input" : "messages";
    CHECK(requests[1][key] == requests[0][key]);

    REQUIRE(client.stream(AgentTest::Request(history), callbacks, response) == Result::SUCCESS);
    REQUIRE(requests.size() == 3);
    CHECK(requests[2] == requests[1]);
}

TEST_CASE("Summary fallback keeps encrypted-content requests and omits unresolvable reasoning ids",
          "[agent][providers][retry]") {
    const bool encrypted = GENERATE(false, true);
    const bool structuredError = GENERATE(false, true);

    AgentTest::ProviderServer provider;
    std::vector<Json> requests;

    Json reasoning{{"type", "reasoning"}, {"id", "rs-one"}, {"summary", Json::array()}};
    if (encrypted) {
        reasoning["encrypted_content"] = "opaque";
    }

    const Json message = {
        {"type", "message"},
        {"id", "msg-one"},
        {"role", "assistant"},
        {"phase", "commentary"},
        {"content", Json::array({{{"type", "output_text"}, {"text", "Checking"}}})},
    };
    const Json call = {
        {"type", "function_call"},
        {"id", "fc-one"},
        {"call_id", "call-one"},
        {"name", "inspect"},
        {"arguments", "{}"},
    };
    const auto output = Json::array({reasoning, message, call});

    provider.server.Post("/v1/responses", [&](const auto& request, auto& response) {
        requests.push_back(Json::parse(request.body));

        if (requests.size() == 1) {
            const char* body = R"({"error":{"message":"reasoning.summary unsupported"}})";
            if (structuredError) {
                body = R"({"error":{"message":"Unsupported summary option","param":"reasoning.summary"}})";
            }
            response.status = 400;
            response.set_content(body, "application/json");
            return;
        }

        if (requests.size() == 2) {
            const Json completed = {{"type", "response.completed"}, {"response", {{"output", output}}}};
            response.set_content("data: " + completed.dump() + "\n\n", "text/event-stream");
            return;
        }

        for (const auto& item : requests.back()["input"]) {
            if (item.value("type", "") == "reasoning" && !item.contains("encrypted_content")) {
                response.status = 400;
                response.set_content(R"({"error":{"message":"Item rs-one not found with store:false"}})",
                                     "application/json");
                return;
            }
        }
        response.set_content(responseCompleted, "text/event-stream");
    });
    provider.start();

    auto config = provider.config(Agent::Provider::Kind::OpenAI);
    UseResponses(config);

    const Agent::Provider client(config);
    const Agent::Provider::Callbacks callbacks{.onReasoning = [](const auto&, bool) {}};
    Agent::Provider::Response response;

    REQUIRE(client.stream(AgentTest::Request({{"user", "Inspect"}}), callbacks, response) == Result::SUCCESS);

    const auto calls = response.toolCalls;
    const auto state = response.state;
    const auto text = response.text;
    REQUIRE(state);
    REQUIRE(calls.size() == 1);

    const std::vector<Agent::Provider::Message> history = {
        {"user", "Inspect"},
        {"assistant", text, calls, "", {}, state},
        {"tool", "{}", {}, "call-one"},
    };
    REQUIRE(client.stream(AgentTest::Request(history), callbacks, response) == Result::SUCCESS);

    const size_t messageIndex = encrypted ? 2 : 1;
    const size_t callIndex = encrypted ? 3 : 2;

    REQUIRE(requests.size() == 3);
    CHECK(requests[1]["include"] == Json::array({"reasoning.encrypted_content"}));
    CHECK(requests[2]["include"] == requests[1]["include"]);
    CHECK_FALSE(requests[2].contains("reasoning"));
    CHECK(requests[2]["input"].size() == (encrypted ? 5 : 4));
    CHECK(requests[2]["input"][messageIndex]["phase"] == "commentary");
    CHECK(requests[2]["input"][messageIndex].contains("id") == encrypted);
    CHECK(requests[2]["input"][callIndex].contains("id") == encrypted);
    CHECK(requests[2]["input"][callIndex]["call_id"] == "call-one");
}

TEST_CASE("Reasoning fallback is limited to one relevant rejection", "[agent][providers][retry]") {
    const bool relevant = GENERATE(false, true);

    AgentTest::ProviderServer provider;
    int count = 0;
    provider.server.Post("/v1/responses", [&](const auto&, auto& response) {
        ++count;

        const char* body = R"({"error":{"message":"Invalid tool schema","param":"tools"}})";
        if (relevant) {
            body = R"({"error":{"message":"Unsupported include option","param":"include"}})";
        }
        response.status = 400;
        response.set_content(body, "application/json");
    });
    provider.start();

    auto config = provider.config();
    UseResponses(config);
    config.retries = 3;

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(config).stream(AgentTest::Request({{"user", "Hello"}}), {}, response) ==
            Result::ERROR);

    const auto& error = response.error;
    CHECK(count == (relevant ? 2 : 1));
    CHECK(error.status == 400);
}

TEST_CASE("Retry budget is bounded and failed attempts do not leak output", "[agent][providers][retry]") {
    const bool recover = GENERATE(false, true);

    AgentTest::ProviderServer provider;
    int count = 0;
    provider.server.Post("/chat", [&](const auto&, auto& response) {
        if (++count > 2 && recover) {
            response.set_content(completion, "text/event-stream");
            return;
        }

        response.status = 503;
        response.set_header("Retry-After", "0");
        response.set_content(R"({"error":{"message":"Busy"}})", "application/json");
    });
    provider.start();

    auto config = provider.config();
    config.retries = 2;

    std::string streamed;
    const Agent::Provider::Callbacks callbacks{.onToken = [&](const auto& token) {
        streamed += token;
    }};

    Agent::Provider::Response response;
    const auto result = Agent::Provider(config).stream(AgentTest::Request({{"user", "Hello"}}),
                                                       callbacks,
                                                       response);

    const auto& text = response.text;
    const auto& error = response.error;
    CHECK(result == (recover ? Result::SUCCESS : Result::ERROR));
    CHECK(count == 3);
    CHECK(text == (recover ? "Done" : ""));
    CHECK(streamed == text);
    CHECK(static_cast<bool>(error) == !recover);
}

TEST_CASE("Streamed text reasoning and tool calls prevent retry even without callbacks",
          "[agent][providers][retry]") {
    const auto delta = GENERATE(
        R"({"content":"Partial"})",
        R"({"reasoning_content":"Thinking"})",
        R"({"tool_calls":[{"index":0,"id":"call-one","function":{"name":"inspect","arguments":"{}"}}]})");

    AgentTest::ProviderServer provider;
    int count = 0;
    provider.server.Post("/chat", [&](const auto&, auto& response) {
        ++count;

        const Json chunk = {{"choices", {{{"delta", Json::parse(delta)}}}}};
        const std::string failure = "\n\ndata: {\"error\":{\"code\":429,\"message\":\"Busy\"}}\n\n";
        response.set_content("data: " + chunk.dump() + failure, "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    config.retries = 3;

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(config).stream(AgentTest::Request({{"user", "Hello"}}), {}, response) ==
            Result::ERROR);
    CHECK(count == 1);
}

TEST_CASE("In-stream errors retry only known transient failures", "[agent][providers][retry]") {
    const auto code = GENERATE("server_error",
                               "rate_limit_exceeded",
                               "invalid_prompt",
                               "context_length_exceeded",
                               "unknown_failure");
    const bool responses = GENERATE(false, true);
    const bool transient = std::string_view(code) == "server_error" ||
                           std::string_view(code) == "rate_limit_exceeded";

    AgentTest::ProviderServer provider;
    int count = 0;
    provider.server.Post(responses ? "/v1/responses" : "/chat", [&](const auto&, auto& response) {
        if (++count != 1) {
            response.set_content(responses ? responseCompleted : completion, "text/event-stream");
            return;
        }

        const Json failure{{"error", {{"code", code}, {"message", "Provider failure"}}}};
        Json event = failure;
        if (responses) {
            event = Json{{"type", "response.failed"}, {"response", failure}};
        }

        response.set_header("Retry-After", "0");
        response.set_content("data: " + event.dump() + "\n\n", "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    if (responses) {
        UseResponses(config);
    }
    config.retries = 3;

    Agent::Provider::Response response;
    const auto result = Agent::Provider(config).stream(AgentTest::Request({{"user", "Hello"}}),
                                                       {},
                                                       response);

    const auto& error = response.error;
    CHECK(result == (transient ? Result::SUCCESS : Result::ERROR));
    CHECK(count == (transient ? 2 : 1));
    if (!transient) {
        CHECK(error.code == code);
    }
}

TEST_CASE("Content filtering and unclassified terminal errors are not retried", "[agent][providers][retry]") {
    const auto event = GENERATE(
        R"({"choices":[{"delta":{},"finish_reason":"content_filter"}]})",
        R"({"choices":[{"delta":{},"finish_reason":"error"}]})",
        R"({"type":"response.incomplete","response":{"incomplete_details":{"reason":"content_filter"}}})");
    const bool responses = Json::parse(event).contains("type");

    AgentTest::ProviderServer provider;
    int count = 0;
    provider.server.Post(responses ? "/v1/responses" : "/chat", [&](const auto&, auto& response) {
        ++count;
        response.set_content(std::string("data: ") + event + "\n\n", "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    if (responses) {
        UseResponses(config);
    }
    config.retries = 3;

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(config).stream(AgentTest::Request({{"user", "Hello"}}), {}, response) ==
            Result::ERROR);
    CHECK(count == 1);
}

TEST_CASE("Cancellation during Retry-After returns cancelled and sends no further request",
          "[agent][providers][retry]") {
    AgentTest::ProviderServer provider;
    std::atomic<int> count{0};
    std::promise<void> entered;
    auto received = entered.get_future();
    provider.server.Post("/chat", [&](const auto&, auto& response) {
        if (++count == 1) {
            entered.set_value();
        }

        response.status = 429;
        response.set_header("Retry-After", "120");
        response.set_content(R"({"error":{"message":"Busy"}})", "application/json");
    });
    provider.start();

    auto config = provider.config();
    config.retries = 3;

    std::atomic<bool> abort{false};
    auto result = std::async(std::launch::async, [&] {
        const Agent::Provider::Callbacks callbacks{.shouldAbort = [&] {
            return abort.load();
        }};

        Agent::Provider::Response response;
        Agent::Provider(config).stream(AgentTest::Request({{"user", "Hello"}}), callbacks, response);
        return response.error;
    });

    const bool started = received.wait_for(3s) == std::future_status::ready;
    std::this_thread::sleep_for(50ms);
    abort.store(true);
    const bool stopped = result.wait_for(1s) == std::future_status::ready;

    REQUIRE(started);
    REQUIRE(stopped);
    CHECK(result.get().kind == Agent::Provider::Error::Kind::Cancelled);
    CHECK(count.load() == 1);
}

TEST_CASE("Output token exhaustion remains an error and is never retried", "[agent][providers][retry]") {
    const bool responses = GENERATE(false, true);

    AgentTest::ProviderServer provider;
    int count = 0;
    provider.server.Post(responses ? "/v1/responses" : "/chat", [&](const auto&, auto& response) {
        ++count;

        const char* stream = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n";
        if (responses) {
            stream = "data: {\"type\":\"response.incomplete\",\"response\":"
                     "{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n";
        }
        response.set_content(stream, "text/event-stream");
    });
    provider.start();

    auto config = provider.config();
    if (responses) {
        UseResponses(config);
    }
    config.retries = 3;

    Agent::Provider::Response response;
    REQUIRE(Agent::Provider(config).stream(AgentTest::Request({{"user", "Hello"}}), {}, response) ==
            Result::ERROR);

    const auto& error = response.error;
    CHECK(count == 1);
    CHECK(error.kind == Agent::Provider::Error::Kind::OutputLimit);
    CHECK(error.message.find("output token limit") != std::string::npos);
    CHECK(response.toolCalls.empty());
    CHECK_FALSE(response.state);
}

#endif
