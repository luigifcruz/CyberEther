#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "compositor/default/actions/agent.hh"
#include "environment.hh"
#include "tools/random.hh"

#include <filesystem>
#include <future>
#include <string>
#include <utility>
#include <vector>

using namespace Jetstream;

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC)

namespace {

struct SettingsScope {
    std::filesystem::path root = std::filesystem::temp_directory_path() /
                                 ("cyberether-provider-settings-" + Random::HexToken());
#if defined(JST_OS_LINUX)
    TestSupport::ScopedEnvironment configHome{"XDG_CONFIG_HOME"};
#else
    TestSupport::ScopedEnvironment configHome{"CFFIXED_USER_HOME"};
#endif

    SettingsScope() {
        REQUIRE(configHome.set(Platform::PathToUtf8(root)));
    }

    ~SettingsScope() {
        Settings::Flush();
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

}  // namespace

TEST_CASE("Provider normalization takes the request format from the matched profile",
          "[agent][compositor][providers][normalize]") {
    using Kind = Agent::Provider::Kind;

    SECTION("a built-in endpoint overrides a stored format that cannot work there") {
        const Settings::Intelligence::Provider stored{
            .name = "OpenAI",
            .host = "https://api.openai.com",
            .api = "chat_completions",
            .chatPath = "/v1/responses",
        };
        REQUIRE(AgentProviders::Identify(stored) == Kind::OpenAI);

        const auto normalized = AgentProviders::Normalize(stored);
        CHECK(normalized.api == "responses");
        CHECK(normalized.chatPath == "/v1/responses");
    }

    SECTION("a built-in endpoint fills in a missing format") {
        const auto normalized = AgentProviders::Normalize({
            .name = "OpenRouter",
            .host = "https://openrouter.ai",
            .chatPath = "/api/v1/chat/completions",
        });
        CHECK(normalized.api == "chat_completions");
        CHECK(normalized.modelsPath == "/api/v1/models");
    }

    SECTION("the Anthropic endpoint selects the Messages format") {
        const Settings::Intelligence::Provider stored{
            .name = "Anthropic",
            .host = "https://api.anthropic.com",
            .chatPath = "/v1/messages",
        };
        REQUIRE(AgentProviders::Identify(stored) == Kind::Anthropic);

        const auto normalized = AgentProviders::Normalize(stored);
        CHECK(normalized.api == "messages");
        CHECK(normalized.modelsPath == "/v1/models?limit=1000");
    }

    SECTION("a custom endpoint keeps the configured format") {
        const Settings::Intelligence::Provider stored{
            .name = "Local",
            .host = "http://localhost:11434",
            .api = "responses",
            .chatPath = "/v1/responses",
        };
        REQUIRE(AgentProviders::Identify(stored) == Kind::Custom);

        CHECK(AgentProviders::Normalize(stored).api == "responses");
    }
}

TEST_CASE("Failed keyring cleanup only permits removing externally managed providers",
          "[agent][compositor][providers][keyring]") {
    using Task = DefaultCompositorState::AgentState::ProviderKeyTask;
    const bool external = GENERATE(false, true);
    const auto kind = GENERATE(Task::Kind::Remove, Task::Kind::Save);
    CAPTURE(external, kind);
    const bool removed = external && kind == Task::Kind::Remove;
    SettingsScope scope;

    const Settings::Intelligence::Provider provider{
        .name = "Test provider",
        .host = "https://example.invalid",
        .apiKeyId = "retained-key-id",
        .apiKeyEnv = external ? "CYBERETHER_TEST_PROVIDER_KEY" : "",
    };
    Settings settings;
    settings.intelligence.customProviders.push_back(provider);
    settings.intelligence.models.push_back({.provider = provider.name, .id = "test-model"});
    settings.intelligence.provider = provider.name;
    settings.intelligence.model = "test-model";
    REQUIRE(Settings::Set(settings) == Result::SUCCESS);

    DefaultCompositorState state;
    state.agent.settings = settings.intelligence;
    state.agent.catalog.models = settings.intelligence.models;
    std::vector<std::pair<Sakura::ToastType, std::string>> notices;
    DefaultCompositorCallbacks callbacks;
    callbacks.notify = [&](auto type, auto, const std::string& message) {
        notices.emplace_back(type, message);
    };
    AgentRuntime runtime(state, callbacks);
    AgentActions actions(state, callbacks, runtime);

    const std::string failure = "Can't remove the API key from the system keychain.";
    std::promise<DefaultCompositorState::AgentState::ProviderKeyResult> cleanup;
    auto& task = state.agent.providers.keyTask;
    task.running = true;
    task.kind = kind;
    task.provider = provider;
    task.previousName = provider.name;
    task.future = cleanup.get_future();
    cleanup.set_value({false, failure});

    actions.reconcile();

    CHECK_FALSE(task.running);
    CHECK_FALSE(task.future.valid());
    REQUIRE(notices.size() == (removed ? 2u : 1u));
    CHECK(notices.front().first == (removed ? Sakura::ToastType::Warning : Sakura::ToastType::Error));
    CHECK(notices.front().second == failure);
    if (removed) {
        CHECK(notices.back().first == Sakura::ToastType::Success);
    }
    CHECK(state.agent.settings.customProviders.empty() == removed);
    CHECK(state.agent.settings.models.empty() == removed);
    CHECK(state.agent.catalog.models.empty() == removed);
    CHECK(state.agent.settings.provider.empty() == removed);

    REQUIRE(Settings::Flush() == Result::SUCCESS);
    Settings persisted;
    REQUIRE(Settings::Get(persisted) == Result::SUCCESS);
    CHECK(persisted.intelligence.customProviders.empty() == removed);
    CHECK(persisted.intelligence.models.empty() == removed);
    if (!removed) {
        REQUIRE(persisted.intelligence.customProviders.size() == 1);
        CHECK(persisted.intelligence.customProviders.front().apiKeyId == provider.apiKeyId);
    }
}

#endif
