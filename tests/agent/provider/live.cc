#include "jetstream/config.hh"

#if defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include "agent_provider.hh"

#include "jetstream/agent/live.hh"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace Jetstream;
using namespace std::chrono_literals;
using Json = nlohmann::json;

namespace {

struct LiveServer : AgentTest::ProviderServer {
    std::mutex mutex;
    std::vector<Json> events;
    std::vector<std::string> outbound;

    LiveServer() {
        server.WebSocket("/v1/live/sessions", [this](const httplib::Request&, httplib::ws::WebSocket& socket) {
            serve(socket);
        });
        start();
    }

    ~LiveServer() {
        stop();
    }

    void serve(httplib::ws::WebSocket& socket) {
        socket.set_read_timeout(20ms);
        while (socket.is_open()) {
            std::string payload;
            const auto result = socket.read(payload);
            flush(socket);
            if (result == httplib::ws::Timeout) {
                continue;
            }
            if (result != httplib::ws::Text) {
                break;
            }

            const auto event = Json::parse(payload, nullptr, false);
            const std::string type = event.is_object() ? event.value("type", "") : "";
            {
                std::lock_guard lock(mutex);
                events.push_back(event);
            }

            if (type == "session.start") {
                socket.send(Json{{"type", "session.started"}}.dump());
            } else if (type == "session.close") {
                socket.send(Json{{"type", "session.closed"}, {"reason", "client"}}.dump());
            }
        }
    }

    void flush(httplib::ws::WebSocket& socket) {
        std::vector<std::string> frames;
        {
            std::lock_guard lock(mutex);
            frames.swap(outbound);
        }
        for (const auto& frame : frames) {
            socket.send(frame);
        }
    }

    void push(const Json& event) {
        std::lock_guard lock(mutex);
        outbound.push_back(event.dump());
    }

    std::size_t size() {
        std::lock_guard lock(mutex);
        return events.size();
    }

    std::vector<Json> since(std::size_t from) {
        std::lock_guard lock(mutex);
        return {events.begin() + static_cast<std::ptrdiff_t>(std::min(from, events.size())), events.end()};
    }

    std::vector<Json> typed(const std::string& type) {
        std::lock_guard lock(mutex);
        std::vector<Json> matches;
        for (const auto& event : events) {
            if (event.value("type", "") == type) {
                matches.push_back(event);
            }
        }
        return matches;
    }

    template<typename Predicate>
    static bool WaitFor(Predicate predicate, std::chrono::milliseconds timeout = 3s) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!predicate() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(5ms);
        }
        return predicate();
    }
};

struct LiveClient {
    std::atomic<Agent::Live::State> state{Agent::Live::State::Idle};
    std::atomic<bool> inputDone{false};
    std::mutex mutex;
    std::vector<bool> delegations;
    std::unique_ptr<Agent::Live> live;

    explicit LiveClient(const LiveServer& server) {
        Agent::Live::Callbacks callbacks;
        callbacks.onState = [this](Agent::Live::State next, const std::string&) {
            state.store(next);
        };
        callbacks.onInputTranscriptDone = [this] {
            inputDone.store(true);
        };
        callbacks.onDelegation = [this](bool active) {
            std::lock_guard lock(mutex);
            delegations.push_back(active);
        };

        live = std::make_unique<Agent::Live>(Agent::Live::Config{
            .provider = std::make_shared<const Agent::Provider>(server.config(Agent::Provider::Kind::OpenAI)),
            .model = "gpt-live-1",
            .backendModel = AgentTest::ProviderServer::modelValue,
            .voice = "marin",
        }, std::move(callbacks));
    }

    std::vector<bool> delegated() {
        std::lock_guard lock(mutex);
        return delegations;
    }
};

std::vector<std::string> Types(const std::vector<Json>& events) {
    std::vector<std::string> types;
    for (const auto& event : events) {
        types.push_back(event.value("type", ""));
    }
    return types;
}

Agent::Live::ToolOutputs Outputs() {
    return {{Agent::Provider::FunctionCall{"call-1", "inspect_flowgraph", "{}"},
             Agent::Tools::Output{R"({"ok":true})"}}};
}

std::vector<Json> Reply(LiveServer& server, Agent::Live& live, std::size_t expected) {
    const auto before = server.size();
    live.reply(Outputs());
    REQUIRE(LiveServer::WaitFor([&] {
        return server.size() >= before + expected;
    }));
    std::this_thread::sleep_for(50ms);
    return server.since(before);
}

}  // namespace

TEST_CASE("Voice sessions greet the user once after the session starts", "[agent][providers][live]") {
    LiveServer server;
    LiveClient client(server);

    REQUIRE(client.live->start() == Result::SUCCESS);
    REQUIRE(LiveServer::WaitFor([&] {
        return client.state.load() == Agent::Live::State::Listening;
    }));
    REQUIRE(LiveServer::WaitFor([&] {
        return !server.typed("session.commentary.append").empty();
    }));
    std::this_thread::sleep_for(50ms);

    const auto events = server.since(0);
    REQUIRE(events.size() >= 2);
    CHECK(events[0]["type"] == "session.start");
    const auto greetings = server.typed("session.commentary.append");
    REQUIRE(greetings.size() == 1);
    CHECK(greetings[0]["delegation_id"].is_null());
    const auto content = greetings[0]["content"].get<std::string>();
    CHECK_FALSE(content.empty());
    CHECK(content.find("English") == std::string::npos);
    CHECK(server.typed("response.create").empty());

    client.live->stop();
    CHECK(client.state.load() == Agent::Live::State::Closed);
}

TEST_CASE("Voice tool replies send outputs and request a response", "[agent][providers][live]") {
    LiveServer server;
    LiveClient client(server);

    REQUIRE(client.live->start() == Result::SUCCESS);
    REQUIRE(LiveServer::WaitFor([&] {
        return !server.typed("session.commentary.append").empty();
    }));

    for (int round = 0; round < 3; ++round) {
        const auto events = Reply(server, *client.live, 2);
        REQUIRE(Types(events) == std::vector<std::string>{"response.item.create", "response.create"});
        CHECK(events[0]["item"]["type"] == "function_call_output");
        CHECK(events[0]["item"]["call_id"] == "call-1");
    }
    CHECK(client.delegated().empty());

    client.live->stop();
}

#endif
