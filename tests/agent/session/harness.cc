#include <catch2/catch_test_macros.hpp>

#include "jetstream/agent/session.hh"
#include "jetstream/config.hh"
#include "tools/random.hh"

#include "agent_voice.hh"

#include <chrono>
#include <thread>
#include <set>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>

#include <nlohmann/json.hpp>

using namespace Jetstream;
using namespace std::chrono_literals;
using Json = nlohmann::json;

namespace {

using Handler = std::function<Result(const Agent::Provider::Request&,
                                     const Agent::Provider::Callbacks&,
                                     Agent::Provider::Response&)>;

struct Client final : Agent::Provider {
    Handler handler;

    explicit Client(Handler handler) : Provider({}), handler(std::move(handler)) {}

    Result stream(const Request& request, const Callbacks& callbacks, Response& response) const override {
        return handler(request, callbacks, response);
    }
};

Agent::Session::Config Config(Handler handler) {
    Agent::Session::Config config;
    config.provider = std::make_shared<Client>(handler);
    config.model = "test";
    config.systemContext = "Inspect before editing; retain the requested sample rate.";
    config.contextLength = 32768;
    config.tools = {{"inspect", "Inspect graph", R"({"type":"object"})"}};
    config.executeTool = [](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{R"({"sample_rate":24000})"};
    };
    return config;
}

bool Wait(Agent::Session& session) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (session.snapshot().streaming && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    return !session.snapshot().streaming;
}

bool Failed(const Agent::Session& session) {
    const auto snapshot = session.snapshot();
    if (snapshot.messages.empty() || snapshot.messages.back().segments.empty()) {
        return false;
    }
    return snapshot.messages.back().segments.back().kind == Agent::Session::Segment::Kind::Error;
}

bool Paired(const std::vector<Agent::Provider::Message>& messages) {
    std::set<std::string> pending;
    for (const auto& message : messages) {
        if (message.role == "tool") {
            if (!pending.erase(message.toolCallId)) {
                return false;
            }
        } else if (!pending.empty()) {
            return false;
        }

        for (const auto& call : message.toolCalls) {
            pending.insert(call.id);
        }
    }
    return pending.empty();
}

}  // namespace

TEST_CASE("Large tool output stays retrievable without flooding model history", "[agent][conversation][budget]") {
    int step = 0;
    std::vector<std::vector<Agent::Provider::Message>> requests;
    const std::string large = std::string(100000, 'x') + "important tail";

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        requests.push_back(request.messages);

        if (step++ == 0) {
            response.toolCalls = {{"inspect-1", "inspect", "{}"}};
        } else if (step == 2) {
            const auto bounded = Json::parse(request.messages.back().content);
            const Json args = {{"id", bounded["result_id"]}, {"offset", 100000}, {"limit", 64}};
            response.toolCalls = {{"read-1", "read_tool_result", args.dump()}};
        } else {
            response.text = "Verified tail.";
        }
        return Result::SUCCESS;
    });
    config.maxToolResultChars = 1024;
    config.executeTool = [&](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{large};
    };

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    REQUIRE(requests.size() == 3);
    const auto& bounded = requests[1].back().content;
    CHECK(bounded.size() <= 1024);
    CHECK(Json::parse(bounded)["truncated"] == true);
    CHECK(Json::parse(requests[2].back().content)["text"] == "important tail");
    CHECK_FALSE(Failed(session));
}

TEST_CASE("Compaction preserves recent requests and complete tool exchanges across user turns",
          "[agent][conversation][compaction]") {
    int calls = 0;
    int summaries = 0;
    std::vector<std::vector<Agent::Provider::Message>> requests;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        requests.push_back(request.messages);

        if (!request.allowTools) {
            ++summaries;
            response.text = "Objective: retain sample rate 24000 Hz. The inspect tool verified it. "
                            "Next: follow the user's latest request.";
        } else if (calls++ == 0) {
            response.toolCalls = {{"inspect-1", "inspect", "{}"}};
        } else if (calls == 2) {
            response.text = std::string(6000, 'a');
        } else {
            response.text = "Done";
        }
        return Result::SUCCESS;
    });
    config.contextLength = 4096;
    config.maxOutputTokens = 512;

    Agent::Session session;
    REQUIRE(session.submit("Keep sample rate 24000 Hz", config));
    REQUIRE(Wait(session));

    const std::string followup(4000, 'b');
    REQUIRE(session.submit(followup, config));
    REQUIRE(Wait(session));

    REQUIRE(summaries == 1);
    REQUIRE_FALSE(Failed(session));

    const auto& compacted = requests.back();
    CHECK(compacted.back().content == followup);
    CHECK(compacted[1].content.find("24000 Hz") != std::string::npos);
    for (const auto& request : requests) {
        CHECK(Paired(request));
    }

    REQUIRE(session.submit("What sample rate did we keep?", config));
    REQUIRE(Wait(session));

    CHECK(requests.back()[1].content.find("24000 Hz") != std::string::npos);
    CHECK(requests.back().back().content == "What sample rate did we keep?");
}

TEST_CASE("Provider context overflow compacts and retries only once", "[agent][conversation][compaction]") {
    int calls = 0;
    int summaries = 0;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        if (!request.allowTools) {
            ++summaries;
            response.text = "Keep 24000 Hz.";
            return Result::SUCCESS;
        }

        if (++calls > 1) {
            response.error = {Agent::Provider::Error::Kind::ContextLength, 400, "Too long"};
            return Result::ERROR;
        }

        response.text = "24000 Hz";
        return Result::SUCCESS;
    });

    Agent::Session session;
    REQUIRE(session.submit("Remember the sample rate", config));
    REQUIRE(Wait(session));
    REQUIRE(session.submit("Continue", config));
    REQUIRE(Wait(session));

    CHECK(calls == 3);
    CHECK(summaries == 1);
    CHECK(Failed(session));
}

TEST_CASE("Interrupted replies are marked incomplete and do not replay opaque completed state",
          "[agent][conversation][history]") {
    int calls = 0;
    std::vector<Agent::Provider::Message> followup;

    auto config = Config([&](const auto& request, const auto& callbacks, auto& response) {
        if (++calls == 1) {
            response.text = "The graph is fixed";
            callbacks.onToken(response.text);

            const Agent::Provider::State state{"host", "path", "test", "opaque"};
            response.state = std::make_shared<Agent::Provider::State>(state);
            response.error = {Agent::Provider::Error::Kind::Connection, 0, "Disconnected"};
            return Result::ERROR;
        }

        followup = request.messages;
        response.text = "Rechecking";
        return Result::SUCCESS;
    });

    Agent::Session session;
    REQUIRE(session.submit("Fix it", config));
    REQUIRE(Wait(session));
    REQUIRE(session.submit("Was it completed?", config));
    REQUIRE(Wait(session));

    REQUIRE(followup.size() == 4);
    CHECK(followup[2].content.starts_with("[Incomplete assistant response"));
    CHECK_FALSE(followup[2].state);
}

TEST_CASE("Compaction includes long message tails and middle constraints",
          "[agent][conversation][compaction]") {
    int calls = 0;
    int summaries = 0;
    std::string source;
    std::vector<Agent::Provider::Message> latest;

    const std::string first = "Keep 24000 Hz. " + std::string(13000, 'x') +
                              "Never delete source. " + std::string(13000, 'y') +
                              "Preserve gain 0.75.";
    const auto description = std::string(7000, 'z') + "Keep the full tool argument tail";
    const auto arguments = Json({{"description", description}}).dump();

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        latest = request.messages;

        if (!request.allowTools) {
            ++summaries;
            source += request.messages.back().content;
            response.text = "Keep 24000 Hz. Never delete source. Preserve gain 0.75.";
        } else if (++calls == 1) {
            response.toolCalls = {{"inspect-long", "inspect", arguments}};
        } else {
            response.text = "Done";
        }
        return Result::SUCCESS;
    });
    config.contextLength = 65536;
    config.maxOutputTokens = 512;

    Agent::Session session;
    REQUIRE(session.submit(first, config));
    REQUIRE(Wait(session));

    config.contextLength = 4096;
    REQUIRE(session.submit("Continue", config));
    REQUIRE(Wait(session));

    CHECK(summaries > 1);
    CHECK(source.find(first) != std::string::npos);
    CHECK(source.find(arguments) != std::string::npos);
    REQUIRE_FALSE(Failed(session));
    REQUIRE(latest.size() >= 3);
    CHECK(latest.back().content == "Continue");
    CHECK(Paired(latest));
    CHECK(latest[1].content.find("Preserve gain 0.75") != std::string::npos);
}

TEST_CASE("Failed multichunk compaction leaves the original context intact",
          "[agent][conversation][compaction]") {
    int summaries = 0;
    std::vector<Agent::Provider::Message> latest;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        latest = request.messages;

        if (!request.allowTools && ++summaries == 2) {
            response.error = {Agent::Provider::Error::Kind::Connection, 0, "Disconnected"};
            return Result::ERROR;
        }

        response.text = "Keep the constraint";
        return Result::SUCCESS;
    });

    const std::string first = std::string(26000, 'x') + "Keep the original constraint";
    config.contextLength = 65536;
    config.maxOutputTokens = 512;

    Agent::Session session;
    REQUIRE(session.submit(first, config));
    REQUIRE(Wait(session));

    config.contextLength = 4096;
    REQUIRE(session.submit("Continue", config));
    REQUIRE(Wait(session));
    REQUIRE(Failed(session));

    config.contextLength = 65536;
    REQUIRE(session.submit("Retry with a larger model", config));
    REQUIRE(Wait(session));

    CHECK(latest[1].content == first);
    CHECK_FALSE(Failed(session));
}

TEST_CASE("Clearing during compaction cannot republish old context", "[agent][conversation][compaction]") {
    std::promise<void> entered;
    std::promise<void> release;
    auto started = entered.get_future();
    auto released = release.get_future().share();
    int calls = 0;
    std::vector<Agent::Provider::Message> latest;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        if (!request.allowTools) {
            entered.set_value();
            released.wait();
            response.text = "Old summary";
            return Result::SUCCESS;
        }

        latest = request.messages;
        if (++calls == 1) {
            response.text = std::string(6000, 'a');
        } else {
            response.text = "Fresh response";
        }
        return Result::SUCCESS;
    });
    config.contextLength = 4096;
    config.maxOutputTokens = 512;

    Agent::Session session;
    REQUIRE(session.submit("Old objective", config));
    REQUIRE(Wait(session));

    REQUIRE(session.submit(std::string(4000, 'b'), config));
    const bool compacting = started.wait_for(3s) == std::future_status::ready;
    session.clear();
    release.set_value();
    REQUIRE(compacting);
    REQUIRE(Wait(session));

    REQUIRE(session.submit("Fresh objective", config));
    REQUIRE(Wait(session));

    REQUIRE(latest.size() == 2);
    CHECK(latest.back().content == "Fresh objective");
}

TEST_CASE("Historical screenshots have a model-facing retention limit", "[agent][conversation][budget]") {
    int calls = 0;
    std::vector<Agent::Provider::Message> latest;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        latest = request.messages;

        if (++calls % 2) {
            response.toolCalls = {{"capture-" + std::to_string(calls), "inspect", "{}"}};
        } else {
            response.text = "Image inspected";
        }
        return Result::SUCCESS;
    });
    config.maxImages = 1;
    config.executeTool = [](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{"{}", {"data:image/png;base64,dGVzdA=="}};
    };

    Agent::Session session;
    for (int turn = 0; turn < 3; ++turn) {
        REQUIRE(session.submit("Inspect current view", config));
        REQUIRE(Wait(session));
    }

    U64 count = 0;
    for (const auto& message : latest) {
        count += message.images.size();
    }
    CHECK(count == 1);
    CHECK(Paired(latest));
}

TEST_CASE("Mode switching cancels active text and rejects late text and voice callbacks",
          "[agent][conversation][mode]") {
    std::promise<void> entered;
    std::promise<void> release;
    auto started = entered.get_future();
    auto released = release.get_future().share();
    int calls = 0;
    std::vector<Agent::Provider::Message> latest;

    auto config = Config([&](const auto& request, const auto& callbacks, auto& response) {
        latest = request.messages;

        if (++calls == 1) {
            entered.set_value();
            released.wait();
            response.text = "Late text";
            callbacks.onToken(response.text);
        } else {
            response.text = "Fresh text";
        }
        return Result::SUCCESS;
    });

    AgentTest::VoiceLink link;
    Agent::Session session(link.factory(), AgentTest::Silent);
    REQUIRE(session.submit("Old text context", config));
    const bool active = started.wait_for(3s) == std::future_status::ready;

    session.listen({.systemContext = "Fresh voice instructions"});
    const bool rejectedWhileRetiring = !session.submit("Too early", config);
    const bool voiceSurvived = session.status().phase != Agent::Session::Phase::Idle;
    release.set_value();

    REQUIRE(active);
    const auto voice = link.connection(0);
    REQUIRE(voice);
    CHECK(rejectedWhileRetiring);
    CHECK(voiceSurvived);

    voice->onInputTranscript("New speech");
    REQUIRE(Wait(session));
    REQUIRE(session.snapshot().messages.size() == 1);
    CHECK(session.snapshot().messages.front().text == "New speech");

    REQUIRE(session.submit("Fresh text request", config));
    voice->onOutputTranscript("Late speech");
    REQUIRE(Wait(session));

    session.poll();
    CHECK(session.status().phase == Agent::Session::Phase::Idle);
    REQUIRE(latest.size() == 2);
    CHECK(latest.back().content == "Fresh text request");
    CHECK(session.snapshot().messages.size() == 2);
}

TEST_CASE("Retained output IDs cannot alias results from a later session", "[agent][conversation][storage]") {
    std::string oldId;

    for (int run = 0; run < 2; ++run) {
        int calls = 0;
        std::vector<Agent::Provider::Message> latest;

        auto config = Config([&](const auto& request, const auto&, auto& response) {
            latest = request.messages;

            if (++calls == 1) {
                response.toolCalls = {{"inspect", "inspect", "{}"}};
            } else if (calls == 2) {
                const auto result = Json::parse(request.messages.back().content);
                const auto id = result.at("result_id").template get<std::string>();
                if (run == 0) {
                    oldId = id;
                    response.text = "Done";
                } else {
                    response.toolCalls = {{"read", "read_tool_result", Json({{"id", oldId}}).dump()}};
                }
            } else {
                response.text = "Result expired";
            }
            return Result::SUCCESS;
        });
        config.executeTool = [](const auto&, const auto&, const auto&) {
            return Agent::Tools::Output{std::string(50000, 'x')};
        };

        Agent::Session session;
        REQUIRE(session.submit("Inspect", config));
        REQUIRE(Wait(session));

        if (run == 1) {
            CHECK(Json::parse(latest.back().content).contains("error"));
        }
    }
}

TEST_CASE("Context errors after streamed reasoning do not restart the model step",
          "[agent][conversation][compaction]") {
    int calls = 0;

    auto config = Config([&](const auto&, const auto& callbacks, auto& response) {
        if (++calls == 1) {
            response.text = "Ready";
            return Result::SUCCESS;
        }

        callbacks.onReasoning("Checking the graph", false);
        response.error = {Agent::Provider::Error::Kind::ContextLength, 400, "Too long"};
        return Result::ERROR;
    });

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));
    REQUIRE(session.submit("Continue", config));
    REQUIRE(Wait(session));

    CHECK(calls == 2);
    CHECK(Failed(session));
}

TEST_CASE("Automatic output budgets use the model allowance instead of an 8192-token cap",
          "[agent][conversation][budget]") {
    std::vector<U64> budgets;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        budgets.push_back(request.maxOutputTokens);
        response.text = "Completed";
        return Result::SUCCESS;
    });
    config.contextLength = 1000000;
    config.modelOutputTokens = 384000;

    Agent::Session session;
    REQUIRE(session.submit("Build the graph", config));
    REQUIRE(Wait(session));
    REQUIRE(budgets.size() == 1);
    CHECK(budgets.back() == 384000);

    config.maxOutputTokens = 8192;
    REQUIRE(session.submit("Use my explicit limit", config));
    REQUIRE(Wait(session));
    CHECK(budgets.back() == 8192);

    config.maxOutputTokens = 500000;
    REQUIRE(session.submit("Respect the model ceiling", config));
    REQUIRE(Wait(session));
    CHECK(budgets.back() == 384000);
}

TEST_CASE("Compaction reserves reasoning output instead of capping the summary generation at 2048",
          "[agent][conversation][compaction]") {
    int calls = 0;
    int summaries = 0;
    U64 summaryBudget = 0;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        if (!request.allowTools) {
            ++summaries;
            summaryBudget = request.maxOutputTokens;
            if (summaryBudget < 16384) {
                response.error = {Agent::Provider::Error::Kind::OutputLimit,
                                  0,
                                  "Thinking exhausted the generation budget"};
                return Result::ERROR;
            }
            response.text = "Keep sample rate 24000 Hz; continue the requested graph edits.";
            return Result::SUCCESS;
        }

        if (++calls == 1) {
            response.text = std::string(30000, 'a');
        } else {
            response.text = "Done";
        }
        return Result::SUCCESS;
    });
    config.contextLength = 65536;
    config.modelOutputTokens = 16384;

    Agent::Session session;
    REQUIRE(session.submit("Keep 24000 Hz. " + std::string(35000, 'b'), config));
    REQUIRE(Wait(session));
    REQUIRE(session.submit(std::string(65000, 'c'), config));
    REQUIRE(Wait(session));

    CHECK(summaries == 1);
    CHECK(summaryBudget == 16384);
    CHECK_FALSE(Failed(session));
}
