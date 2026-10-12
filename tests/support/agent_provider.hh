#ifndef JETSTREAM_TESTS_SUPPORT_AGENT_PROVIDER_HH
#define JETSTREAM_TESTS_SUPPORT_AGENT_PROVIDER_HH

#include "jetstream/agent/provider.hh"

#include <catch2/catch_test_macros.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <span>
#include <thread>
#include <vector>

namespace AgentTest {

namespace Sse {

inline std::string Event(const nlohmann::json& event) {
    return "data: " + event.dump() + "\n\n";
}

inline std::string Delta(const nlohmann::json& delta) {
    return Event({{"choices", {{{"delta", delta}}}}});
}

inline std::string Done() {
    return "data: [DONE]\n\n";
}

}  // namespace Sse

struct ProviderServer {
    static constexpr const char* keyValue = "local-agent-test-key";
    static constexpr const char* modelValue = "test-model";

    httplib::Server server;
    std::thread listener;
    std::string origin;

    void start() {
        const int port = server.bind_to_any_port("127.0.0.1");
        REQUIRE(port > 0);
        origin = "http://127.0.0.1:" + std::to_string(port);
        listener = std::thread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }

    void stop() {
        server.stop();
        if (listener.joinable()) {
            listener.join();
        }
    }

    ~ProviderServer() {
        stop();
    }

    Jetstream::Agent::Provider::Config config(
            Jetstream::Agent::Provider::Kind kind = Jetstream::Agent::Provider::Kind::Custom) const {
        return {
            .name = "Test",
            .kind = kind,
            .host = origin,
            .path = "/chat",
            .modelsPath = "/models",
            .apiKey = [] {
                return std::string(keyValue);
            },
            .timeoutSeconds = 2,
            .requestReasoning = true,
        };
    }
};

inline Jetstream::Agent::Provider::Request Request(const std::vector<Jetstream::Agent::Provider::Message>& messages,
                                                   std::span<const Jetstream::Agent::Provider::Function> tools = {}) {
    return {.messages = messages, .tools = tools, .model = ProviderServer::modelValue};
}

}  // namespace AgentTest

#endif  // JETSTREAM_TESTS_SUPPORT_AGENT_PROVIDER_HH
