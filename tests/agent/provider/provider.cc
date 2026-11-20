#include <catch2/catch_test_macros.hpp>

#include "jetstream/agent/provider.hh"
#include "tools/random.hh"

using namespace Jetstream;

TEST_CASE("Agent session identifiers are unique and opaque", "[agent][provider]") {
    const auto first = Random::HexToken();
    const auto second = Random::HexToken();

    REQUIRE(first.size() == 32);
    REQUIRE(second.size() == 32);
    REQUIRE(first != second);
    REQUIRE(first.find_first_not_of("0123456789abcdef") == std::string::npos);
}

TEST_CASE("Provider detection matches exact endpoints rather than host substrings",
          "[agent][providers]") {
    using Kind = Agent::Provider::Kind;
    const auto identify = Agent::Provider::Profile::Identify;
    REQUIRE(identify("https://api.openai.com", "/v1/chat/completions") ==
            Kind::Custom);
    REQUIRE(identify("https://api.openai.com", "/v1/responses") ==
            Kind::OpenAI);
    REQUIRE(identify("https://OPENROUTER.AI/", "/api/v1/chat/completions") ==
            Kind::OpenRouter);
    REQUIRE(identify("https://opencode.ai", "/zen/go/v1/chat/completions") ==
            Kind::OpenCodeGo);
    REQUIRE(identify("https://openrouter.ai.example.com", "") ==
            Kind::Custom);
    REQUIRE(identify("https://example.com/openrouter.ai", "") ==
            Kind::Custom);
    REQUIRE(identify("http://localhost:11434", "/v1/chat/completions") ==
            Kind::Custom);
    REQUIRE(identify("https://opencode.ai", "/different-api") ==
            Kind::Custom);
    REQUIRE(identify("https://api.anthropic.com/", "/v1/messages") ==
            Kind::Anthropic);
    REQUIRE(identify("https://api.anthropic.com", "/v1/chat/completions") ==
            Kind::Custom);
}

TEST_CASE("Anthropic profile uses the Messages API", "[agent][providers]") {
    using Api = Agent::Provider::Api;
    const auto& profile = Agent::Provider::Profile::Find(Agent::Provider::Kind::Anthropic);
    REQUIRE(profile.kind == Agent::Provider::Kind::Anthropic);
    CHECK(profile.api == Api::Messages);
    CHECK(profile.chatPath == "/v1/messages");
    CHECK(profile.modelsPath == "/v1/models");
    CHECK(profile.catalogId == "anthropic");
    CHECK(profile.requestReasoning);
    CHECK_FALSE(profile.ready(Agent::Provider::Scope::Chat, false));
    CHECK(Agent::Provider::Profile::All().back().kind == Agent::Provider::Kind::Custom);

    for (const auto api : {Api::ChatCompletions, Api::Responses, Api::Messages}) {
        CHECK(Agent::Provider::Parse(Agent::Provider::Name(api), Api::ChatCompletions) == api);
    }
    CHECK(std::string(Agent::Provider::Name(Api::Messages)) == "messages");
}
