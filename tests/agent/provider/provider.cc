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
}
