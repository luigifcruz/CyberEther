#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

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

U64 Images(const std::vector<Agent::Provider::Message>& messages) {
    U64 count = 0;
    for (const auto& message : messages) {
        count += message.images.size();
    }
    return count;
}

std::string LastError(const Agent::Session& session) {
    const auto snapshot = session.snapshot();
    if (snapshot.messages.empty() || snapshot.messages.back().segments.empty()) {
        return {};
    }
    const auto& segment = snapshot.messages.back().segments.back();
    return segment.kind == Agent::Session::Segment::Kind::Error ? segment.text : std::string{};
}

bool Mentions(const std::vector<Agent::Provider::Message>& messages, const std::string& text) {
    for (const auto& message : messages) {
        if (message.content.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

const std::string kImage = "data:image/png;base64,dGVzdA==";

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

TEST_CASE("Final budget response retains tool schemas while disabling execution", "[agent][conversation][budget]") {
    int calls = 0;
    int executions = 0;
    bool toolsRetained = false;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        if (++calls == 1) {
            response.toolCalls = {{"call", "inspect", "{}"}};
        } else {
            toolsRetained = !request.allowTools && request.tools.size() == 2;
            response.text = "Budget reached.";
        }
        return Result::SUCCESS;
    });
    config.maxToolTurns = 1;
    config.executeTool = [&](const auto&, const auto&, const auto&) {
        ++executions;
        return Agent::Tools::Output{"{}"};
    };

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    CHECK(toolsRetained);
    CHECK(executions == 0);
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

TEST_CASE("Tool images ride on the tool message when enabled", "[agent][conversation][images]") {
    int calls = 0;
    std::vector<std::vector<Agent::Provider::Message>> requests;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        requests.push_back(request.messages);
        if (++calls == 1) {
            response.toolCalls = {{"capture", "inspect", "{}"}};
        } else {
            response.text = "Inspected";
        }
        return Result::SUCCESS;
    });
    config.toolResultImages = true;
    config.executeTool = [](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{R"({"value":42})", {kImage}};
    };

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    REQUIRE(requests.size() == 2);
    REQUIRE(requests[1].size() == 4);
    const auto& tool = requests[1][3];
    CHECK(tool.role == "tool");
    CHECK(tool.toolCallId == "capture");
    CHECK(tool.content == R"({"value":42})");
    REQUIRE(tool.images.size() == 1);
    CHECK(tool.images.front() == kImage);
    CHECK(Images(requests[1]) == 1);
    CHECK(Paired(requests[1]));

    REQUIRE(session.submit("Again", config));
    REQUIRE(Wait(session));
    REQUIRE(requests.size() == 3);
    CHECK(requests[2][3].images.size() == 1);

    SECTION("a provider without the option receives no tool-message images") {
        config.toolResultImages = false;
        REQUIRE(session.submit("Once more", config));
        REQUIRE(Wait(session));

        for (const auto& message : requests.back()) {
            CHECK((message.role != "tool" || message.images.empty()));
        }
        CHECK(Mentions(requests.back(), "does not accept images in tool results"));
    }
}

TEST_CASE("A cap of one image holds for two captures in one round", "[agent][conversation][images]") {
    const bool toolImages = GENERATE(false, true);
    int calls = 0;
    std::vector<Agent::Provider::Message> latest;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        latest = request.messages;
        if (++calls == 1) {
            response.toolCalls = {{"first", "inspect", "{}"}, {"second", "inspect", "{}"}};
        } else {
            response.text = "Compared";
        }
        return Result::SUCCESS;
    });
    config.maxImages = 1;
    config.toolResultImages = toolImages;
    config.executeTool = [](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{"{}", {kImage}};
    };

    Agent::Session session;
    REQUIRE(session.submit("Compare both views", config));
    REQUIRE(Wait(session));

    CHECK(Images(latest) == 1);
    CHECK(latest.back().images.size() == 1);
    CHECK(Mentions(latest, "only the newest 1 image(s) are kept"));
    CHECK(Paired(latest));
    CHECK_FALSE(Failed(session));
}

TEST_CASE("An image-limit rejection lowers the cap, retries once, and sticks for the conversation",
          "[agent][conversation][images]") {
    int rejections = 0;
    int turnCalls = 0;
    std::vector<U64> sent;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        sent.push_back(Images(request.messages));
        if (sent.back() > 1) {
            ++rejections;
            response.error = {Agent::Provider::Error::Kind::InvalidRequest, 400,
                              "At most 1 image(s) may be provided in one prompt. (parameter=image)"};
            response.error.imageLimit = 1;
            return Result::ERROR;
        }

        if (++turnCalls == 1) {
            response.toolCalls = {{"first", "inspect", "{}"}, {"second", "inspect", "{}"}};
        } else {
            response.text = "Compared";
        }
        return Result::SUCCESS;
    });
    config.executeTool = [](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{"{}", {kImage}};
    };

    Agent::Session session;
    REQUIRE(session.submit("Compare both views", config));
    REQUIRE(Wait(session));

    CHECK(rejections == 1);
    CHECK(sent == std::vector<U64>{0, 2, 1});
    CHECK_FALSE(Failed(session));

    bool announced = false;
    const auto snapshot = session.snapshot();
    for (const auto& segment : snapshot.messages.back().segments) {
        announced = announced || segment.text.find("at most 1 image(s) per request") != std::string::npos;
    }
    CHECK(announced);

    turnCalls = 0;
    REQUIRE(session.submit("Compare again", config));
    REQUIRE(Wait(session));
    CHECK(rejections == 1);
    CHECK_FALSE(Failed(session));

    SECTION("clearing the conversation forgets the learned cap") {
        session.clear();
        turnCalls = 0;
        REQUIRE(session.submit("Compare both views", config));
        REQUIRE(Wait(session));
        CHECK(rejections == 2);
        CHECK_FALSE(Failed(session));
    }
}

TEST_CASE("An image-limit rejection the cap already satisfies is not retried", "[agent][conversation][images]") {
    int calls = 0;

    auto config = Config([&](const auto&, const auto&, auto& response) {
        ++calls;
        response.error = {Agent::Provider::Error::Kind::InvalidRequest, 400, "At most 1 image(s) may be provided."};
        response.error.imageLimit = 1;
        return Result::ERROR;
    });
    config.maxImages = 1;

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    CHECK(calls == 1);
    CHECK(Failed(session));
}

TEST_CASE("A blank reply after the tool-step budget is retried with a plain-text instruction",
          "[agent][conversation][budget]") {
    const bool recovers = GENERATE(true, false);
    int calls = 0;
    std::vector<std::vector<Agent::Provider::Message>> requests;
    std::vector<bool> allowed;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        requests.push_back(request.messages);
        allowed.push_back(request.allowTools);
        if (++calls == 1) {
            response.toolCalls = {{"call", "inspect", "{}"}};
        } else if (calls == 2 || !recovers) {
            response.text = request.allowTools ? "Later answer." : "\n\n";
        } else {
            response.text = "Summary.";
        }
        return Result::SUCCESS;
    });
    config.maxToolTurns = 1;

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    REQUIRE(requests.size() == 3);
    CHECK_FALSE(allowed[1]);
    CHECK_FALSE(allowed[2]);
    CHECK(requests[2].back().role == "user");
    CHECK(requests[2].back().content.find("Tools are now disabled") != std::string::npos);
    CHECK(requests[2].size() == requests[1].size() + 1);

    if (recovers) {
        CHECK_FALSE(Failed(session));
    } else {
        CHECK(Failed(session));
        CHECK(LastError(session).find("empty reply instead of a summary") != std::string::npos);
    }

    REQUIRE(session.submit("Continue", config));
    REQUIRE(Wait(session));
    CHECK_FALSE(Mentions(requests.back(), "Tools are now disabled"));
    CHECK_FALSE(Mentions(requests.back(), "[Incomplete assistant response"));
    CHECK(Mentions(requests.back(), "Summary.") == recovers);
    for (const auto& message : requests.back()) {
        CHECK((message.role != "assistant" || !message.toolCalls.empty() ||
               message.content.find_first_not_of(" \n") != std::string::npos));
    }
}

TEST_CASE("A blank reply in an ordinary round is retried once", "[agent][conversation]") {
    const bool recovers = GENERATE(true, false);
    int calls = 0;
    int executions = 0;
    std::vector<std::vector<Agent::Provider::Message>> requests;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        requests.push_back(request.messages);
        ++calls;
        if (!recovers || calls == 1) {
            response.text = "";
        } else if (calls == 2) {
            response.toolCalls = {{"call", "inspect", "{}"}};
        } else {
            response.text = "Done";
        }
        return Result::SUCCESS;
    });
    config.executeTool = [&](const auto&, const auto&, const auto&) {
        ++executions;
        return Agent::Tools::Output{"{}"};
    };

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    REQUIRE(requests.size() >= 2);
    CHECK(requests[1].back().content.find("Your previous reply was empty") != std::string::npos);

    if (recovers) {
        REQUIRE(requests.size() == 3);
        CHECK(executions == 1);
        CHECK_FALSE(Mentions(requests[2], "Your previous reply was empty"));
        CHECK_FALSE(Failed(session));
    } else {
        CHECK(requests.size() == 2);
        CHECK(Failed(session));
        CHECK(LastError(session).find("empty reply twice") != std::string::npos);
    }
}

TEST_CASE("A cancelled turn does not retry a blank reply", "[agent][conversation]") {
    int calls = 0;
    Agent::Session session;

    auto config = Config([&](const auto&, const auto&, auto& response) {
        ++calls;
        session.cancel();
        response.text = "";
        return Result::SUCCESS;
    });

    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));
    CHECK(calls == 1);
}

TEST_CASE("Blank compaction summaries are retried and never replace history", "[agent][conversation][compaction]") {
    const bool recovers = GENERATE(true, false);
    int calls = 0;
    int summaries = 0;
    std::string retried;
    std::vector<Agent::Provider::Message> latest;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        if (!request.allowTools) {
            if (++summaries == 1 || !recovers) {
                if (summaries == 2) {
                    retried = request.messages.back().content;
                }
                response.text = "\n\n";
            } else {
                retried = request.messages.back().content;
                response.text = "Keep sample rate 24000 Hz.";
            }
            return Result::SUCCESS;
        }

        latest = request.messages;
        response.text = ++calls == 1 ? std::string(30000, 'a') : "Done";
        return Result::SUCCESS;
    });
    config.contextLength = 65536;
    config.modelOutputTokens = 16384;

    const std::string first = "Keep 24000 Hz. " + std::string(35000, 'b');

    Agent::Session session;
    REQUIRE(session.submit(first, config));
    REQUIRE(Wait(session));
    REQUIRE(session.submit(std::string(65000, 'c'), config));
    REQUIRE(Wait(session));

    CHECK(summaries == 2);
    CHECK(retried.ends_with("Write the checkpoint summary now in plain text.]"));
    if (recovers) {
        CHECK_FALSE(Failed(session));
        CHECK(Mentions(latest, "Keep sample rate 24000 Hz."));
    } else {
        CHECK(Failed(session));
        CHECK(LastError(session).find("empty conversation summary twice") != std::string::npos);

        config.contextLength = 1048576;
        REQUIRE(session.submit("Retry with a larger model", config));
        REQUIRE(Wait(session));
        CHECK(summaries == 2);
        REQUIRE(latest.size() >= 3);
        CHECK(latest[1].content == first);
        CHECK(latest[2].content == std::string(30000, 'a'));
        CHECK_FALSE(Failed(session));
    }
}

TEST_CASE("Repeated identical failing tool calls are flagged and then stop the turn",
          "[agent][conversation][tools]") {
    int calls = 0;
    int executions = 0;
    std::vector<std::vector<Agent::Provider::Message>> requests;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        requests.push_back(request.messages);
        response.toolCalls = {{"call-" + std::to_string(++calls), "inspect", R"({"bad":true})"}};
        return Result::SUCCESS;
    });
    config.executeTool = [&](const auto&, const auto&, const auto&) {
        ++executions;
        return Agent::Tools::Output{R"({"error":"nope"})", {}, true};
    };

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    CHECK(executions == 3);
    REQUIRE(requests.size() == 3);
    CHECK(requests[1].back().content == R"({"error":"nope"})");
    CHECK(requests[2].back().content.starts_with(R"({"error":"nope"})"));
    CHECK(requests[2].back().content.find("[Repeated failing call: the tool received exactly the same arguments") !=
          std::string::npos);
    CHECK(Failed(session));
    CHECK(LastError(session).find("same failing 'inspect' tool call 3 times") != std::string::npos);
    CHECK(LastError(session).find("nope") != std::string::npos);

    calls = 100;
    config.executeTool = [](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{"{}"};
    };
    config.maxToolTurns = 1;
    REQUIRE(session.submit("Continue", config));
    REQUIRE(Wait(session));
    CHECK(Paired(requests[3]));
    CHECK(Mentions(requests[3], "third identical failure"));
}

TEST_CASE("Repeats that succeed, differ, or hit the budget are not flagged", "[agent][conversation][tools]") {
    int calls = 0;
    int executions = 0;
    std::vector<Agent::Provider::Message> latest;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        latest = request.messages;
        switch (++calls) {
            case 1: response.toolCalls = {{"a", "inspect", R"({"a":1})"}}; break;
            case 2: response.toolCalls = {{"b", "inspect", R"({"a":2})"}}; break;
            case 3: response.toolCalls = {{"c", "inspect", R"({"a":1})"}}; break;
            case 4: response.toolCalls = {{"d", "inspect", R"({"a":1})"}}; break;
            default: response.text = "Done"; break;
        }
        return Result::SUCCESS;
    });
    config.maxToolTurns = 4;
    config.executeTool = [&](const auto&, const auto&, const auto&) {
        return ++executions < 3 ? Agent::Tools::Output{R"({"error":"nope"})", {}, true}
                                : Agent::Tools::Output{"{}"};
    };

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    CHECK(calls == 5);
    CHECK_FALSE(Mentions(latest, "Repeated failing call"));
    CHECK(Mentions(latest, "tool-step budget reached"));
    CHECK_FALSE(Failed(session));
}

TEST_CASE("Output-limit advice matches the configured allowance", "[agent][conversation][budget]") {
    const std::string cutoff = "The response reached the output token limit and was cut off.";
    Agent::Provider::Usage usage;
    std::string text;

    auto config = Config([&](const auto&, const auto& callbacks, auto& response) {
        response.usage = usage;
        response.text = text;
        if (!text.empty() && callbacks.onToken) {
            callbacks.onToken(text);
        }
        response.error = {Agent::Provider::Error::Kind::OutputLimit, 0, cutoff};
        return Result::ERROR;
    });
    config.contextLength = 1048576;

    Agent::Session session;

    SECTION("the automatic allowance is never suggested when it is already in use") {
        REQUIRE(session.submit("Go", config));
        REQUIRE(Wait(session));
        CHECK(LastError(session) == cutoff + " Reasoning and the answer share this budget. maxOutputTokens is 0 "
                                             "(automatic), which allowed 32768 tokens; set it above 32768 to "
                                             "allow more, then continue the task.");
    }

    SECTION("a budget spent entirely on reasoning is reported") {
        usage.outputTokens = 32768;
        usage.reasoningTokens = 32768;
        REQUIRE(session.submit("Go", config));
        REQUIRE(Wait(session));
        CHECK(LastError(session).find("All 32768 output tokens went to reasoning and no answer was produced") !=
              std::string::npos);
        CHECK(LastError(session).find("set it above 32768") != std::string::npos);
    }

    SECTION("partial text is not reported as reasoning-only") {
        usage.outputTokens = 100;
        usage.reasoningTokens = 100;
        text = "Partial";
        REQUIRE(session.submit("Go", config));
        REQUIRE(Wait(session));
        CHECK(LastError(session).find("went to reasoning") == std::string::npos);
        CHECK(LastError(session).find("Reasoning and the answer share this budget.") != std::string::npos);
    }

    SECTION("the model ceiling asks for a smaller step") {
        config.contextLength = 65536;
        config.modelOutputTokens = 16384;
        REQUIRE(session.submit("Go", config));
        REQUIRE(Wait(session));
        CHECK(LastError(session).ends_with("maxOutputTokens is 0 (automatic) and 16384 tokens is the most this "
                                           "model allows; ask for a smaller step, then continue the task."));
    }

    SECTION("an explicit limit keeps the existing advice") {
        config.maxOutputTokens = 512;
        REQUIRE(session.submit("Go", config));
        REQUIRE(Wait(session));
        CHECK(LastError(session).ends_with("Increase maxOutputTokens or set it to 0 for the automatic model "
                                           "allowance, then continue the task."));
    }
}

TEST_CASE("Identical failing calls in one reply count once and the stopping round still completes",
          "[agent][conversation][tools]") {
    int calls = 0;
    std::vector<std::string> executed;
    std::vector<std::vector<Agent::Provider::Message>> requests;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        requests.push_back(request.messages);
        const std::string round = std::to_string(++calls);
        if (calls == 1) {
            response.toolCalls = {{"a1", "inspect", R"({"bad":true})"},
                                  {"b1", "inspect", R"({"bad":true})"},
                                  {"c1", "inspect", R"({"bad":true})"}};
        } else if (calls == 2) {
            response.toolCalls = {{"a2", "inspect", R"({"bad":true})"}};
        } else if (calls == 3) {
            response.toolCalls = {{"a3", "inspect", R"({"bad":true})"}, {"b3", "inspect", R"({"good":true})"}};
        } else {
            response.text = "Done";
        }
        return Result::SUCCESS;
    });
    config.executeTool = [&](const auto&, const std::string& args, const auto&) {
        executed.push_back(args);
        return args.find("bad") != std::string::npos ? Agent::Tools::Output{R"({"error":"nope"})", {}, true}
                                                      : Agent::Tools::Output{R"({"ok":true})"};
    };

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    REQUIRE(requests.size() == 3);
    CHECK_FALSE(Mentions(requests[1], "Repeated failing call"));
    CHECK(Mentions(requests[2], "Resending it unchanged cannot succeed"));
    CHECK(Failed(session));
    REQUIRE(executed.size() == 6);
    CHECK(executed.back() == R"({"good":true})");

    config.maxToolTurns = 1;
    REQUIRE(session.submit("Continue", config));
    REQUIRE(Wait(session));
    CHECK(Paired(requests.back()));
    CHECK(Mentions(requests.back(), R"({"ok":true})"));
    CHECK(Mentions(requests.back(), "third identical failure"));
    CHECK_FALSE(Mentions(requests.back(), "did not complete"));
}

TEST_CASE("The model's extra request fields reach ordinary and summary requests",
          "[agent][conversation][compaction]") {
    int calls = 0;
    std::vector<std::string> normal;
    std::vector<std::string> summary;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        if (!request.allowTools) {
            summary.push_back(request.extraBody);
            response.text = "Keep sample rate 24000 Hz.";
            return Result::SUCCESS;
        }

        normal.push_back(request.extraBody);
        response.text = ++calls == 1 ? std::string(30000, 'a') : "Done";
        return Result::SUCCESS;
    });
    config.contextLength = 65536;
    config.modelOutputTokens = 16384;
    config.extraBody = R"({"chat_template_kwargs":{"enable_thinking":false}})";

    Agent::Session session;
    REQUIRE(session.submit("Keep 24000 Hz. " + std::string(35000, 'b'), config));
    REQUIRE(Wait(session));
    REQUIRE(session.submit(std::string(65000, 'c'), config));
    REQUIRE(Wait(session));

    REQUIRE(normal.size() == 2);
    REQUIRE(summary.size() == 1);
    CHECK(normal[0] == config.extraBody);
    CHECK(normal[1] == config.extraBody);
    CHECK(summary[0] == config.extraBody);
}

TEST_CASE("A learned image cap applies only to the endpoint and model it came from",
          "[agent][conversation][images]") {
    int turnCalls = 0;
    std::vector<U64> sent;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        sent.push_back(Images(request.messages));
        if (request.model == "strict" && sent.back() > 1) {
            response.error = {Agent::Provider::Error::Kind::InvalidRequest, 400, "At most 1 image(s) may be provided."};
            response.error.imageLimit = 1;
            return Result::ERROR;
        }

        if (++turnCalls == 1) {
            response.toolCalls = {{"first", "inspect", "{}"}, {"second", "inspect", "{}"}};
        } else {
            response.text = "Compared";
        }
        return Result::SUCCESS;
    });
    config.model = "strict";
    config.executeTool = [](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{"{}", {kImage}};
    };

    Agent::Session session;
    REQUIRE(session.submit("Compare both views", config));
    REQUIRE(Wait(session));
    CHECK(sent == std::vector<U64>{0, 2, 1});

    config.model = "roomy";
    turnCalls = 0;
    sent.clear();
    REQUIRE(session.submit("Compare again", config));
    REQUIRE(Wait(session));
    CHECK(sent == std::vector<U64>{1, 3});
    CHECK_FALSE(Failed(session));
}

TEST_CASE("A provider configured for no images gets a note that does not invite recapture",
          "[agent][conversation][images]") {
    int calls = 0;
    std::vector<Agent::Provider::Message> latest;

    auto config = Config([&](const auto& request, const auto&, auto& response) {
        latest = request.messages;
        if (++calls == 1) {
            response.toolCalls = {{"capture", "inspect", "{}"}};
        } else {
            response.text = "Done";
        }
        return Result::SUCCESS;
    });
    config.maxImages = 0;
    config.executeTool = [](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{"{}", {kImage}};
    };

    Agent::Session session;
    REQUIRE(session.submit("Inspect", config));
    REQUIRE(Wait(session));

    CHECK(Images(latest) == 0);
    CHECK(Mentions(latest, "configured to receive no images"));
    CHECK_FALSE(Mentions(latest, "capture it again"));
}
