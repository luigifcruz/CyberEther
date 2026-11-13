#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "compositor/default/actions/agent.hh"
#include "environment.hh"
#include "tools/random.hh"

#include <nlohmann/json.hpp>

#include <algorithm>
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

TEST_CASE("Provider image options and model request fields reach the session config",
          "[agent][compositor][providers][images]") {
    DefaultCompositorState state;
    auto& settings = state.agent.settings;
    settings.customProviders.push_back({
        .name = "Local",
        .host = "http://localhost:8000",
        .maxImages = 1,
        .toolResultImages = true,
    });
    settings.models.push_back({
        .provider = "Local",
        .id = "m",
        .contextLength = 1048576,
        .reasoningEfforts = {"none", "medium", "high"},
        .maxContextTokens = 262144,
        .extraBody = R"({"temperature":0.6})",
        .effortBodies = R"({"medium":{"chat_template_kwargs":{"medium_effort":true}}})",
    });
    settings.provider = "Local";
    settings.model = "m";

    const auto chat = AgentRuntime::Chat(state.agent, {});
    CHECK(chat.maxImages == 1);
    CHECK(chat.toolResultImages);
    CHECK(chat.contextLength == 262144);
    CHECK(chat.reasoningEffort == "medium");
    CHECK(nlohmann::json::parse(chat.extraBody) ==
          nlohmann::json({{"temperature", 0.6}, {"chat_template_kwargs", {{"medium_effort", true}}}}));

    const auto call = AgentRuntime::Call(state.agent, {.backend = {.provider = "Local", .id = "m"}}, {});
    CHECK(call.maxImages == 1);
    CHECK(call.toolResultImages);

    SECTION("an unknown provider keeps the session defaults") {
        settings.provider = "Missing";
        const auto fallback = AgentRuntime::Chat(state.agent, {});
        CHECK(fallback.maxImages == Agent::Session::Config{}.maxImages);
        CHECK_FALSE(fallback.toolResultImages);
        CHECK(fallback.extraBody.empty());
    }

    SECTION("defaults match the session and normalization leaves the options alone") {
        CHECK(Settings::Intelligence::Provider{}.maxImages == Agent::Session::Config{}.maxImages);
        CHECK_FALSE(Settings::Intelligence::Provider{}.toolResultImages);

        const auto normalized = AgentProviders::Normalize(settings.customProviders.front());
        CHECK(normalized.maxImages == 1);
        CHECK(normalized.toolResultImages);
    }

    SECTION("a context cap never raises the discovered window") {
        settings.models.front().maxContextTokens = 4194304;
        CHECK(AgentRuntime::Chat(state.agent, {}).contextLength == 1048576);
        CHECK(AgentProviders::ContextWindow(settings.models.front()) == 1048576);
    }

    SECTION("the cap alone bounds a model with no discovered window") {
        settings.models.front().contextLength = 0;
        settings.models.front().maxContextTokens = 262144;
        CHECK(AgentRuntime::Chat(state.agent, {}).contextLength == 262144);
        CHECK(AgentProviders::ContextWindow(settings.models.front()) == 262144);
    }

    SECTION("a provider that takes no images is not offered the capture tool") {
        state.agent.captureSupported = true;
        const auto offers = [](const Agent::Session::Config& config) {
            return std::any_of(config.tools.begin(), config.tools.end(), [](const auto& tool) {
                return tool.name == "inspect_block_view";
            });
        };
        CHECK(offers(AgentRuntime::Chat(state.agent, {})));

        settings.customProviders.front().maxImages = 0;
        CHECK_FALSE(offers(AgentRuntime::Chat(state.agent, {})));
    }
}

TEST_CASE("Chat transcript settings reach the session config", "[agent][compositor][transcript]") {
    SettingsScope scope;
    DefaultCompositorState state;
    auto& settings = state.agent.settings;

    SECTION("transcripts are on by default and live beside the settings file") {
#if defined(JST_OS_LINUX)
        const auto expected = Platform::PathToUtf8(scope.root / "cyberether" / "transcripts");
#else
        std::string configPath;
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        const auto expected = Platform::PathToUtf8(Platform::PathFromUtf8(configPath) / "transcripts");
#endif

        const auto chat = AgentRuntime::Chat(state.agent, {});
        CHECK(chat.transcript.enabled);
        CHECK(chat.transcript.folder == expected);
        CHECK_FALSE(std::filesystem::exists(Platform::PathFromUtf8(expected)));
    }

    SECTION("an absolute folder is passed through untouched") {
        settings.transcriptFolder = "/var/tmp/agent transcripts";
        const auto chat = AgentRuntime::Chat(state.agent, {});
        CHECK(chat.transcript.enabled);
        CHECK(chat.transcript.folder == "/var/tmp/agent transcripts");
    }

    SECTION("a hand-typed folder never depends on where the app was launched") {
        const auto home = scope.root / "home";
        TestSupport::ScopedEnvironment homeVariable{"HOME"};
        REQUIRE(homeVariable.set(Platform::PathToUtf8(home)));

        std::string configPath;
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        const auto config = Platform::PathFromUtf8(configPath);

        const auto folder = [&](const std::string& chosen) {
            settings.transcriptFolder = chosen;
            return Platform::PathFromUtf8(AgentRuntime::Chat(state.agent, {}).transcript.folder);
        };

        CHECK(folder("~/agent-logs/nemotron") == home / "agent-logs" / "nemotron");
        CHECK(folder("~") == home);
        CHECK(folder("~/") == home);
        CHECK(folder("agent-logs") == config / "agent-logs");
        CHECK(folder("./logs/agent") == config / "." / "logs" / "agent");
        CHECK(folder("~someone/logs") == config / "~someone" / "logs");
        CHECK(folder("~/agent-logs").is_absolute());
        CHECK(folder("agent-logs").is_absolute());

        // Resolving the folder creates nothing; the writer does that when a chat starts.
        CHECK_FALSE(std::filesystem::exists(home));

        SECTION("without a home folder the default is used") {
            REQUIRE(homeVariable.set(std::nullopt));
            CHECK(folder("~/agent-logs") == config / "transcripts");
        }
    }

    SECTION("turning transcripts off leaves the session without a folder") {
        settings.transcripts = false;
        settings.transcriptFolder = "/var/tmp/agent transcripts";
        const auto chat = AgentRuntime::Chat(state.agent, {});
        CHECK_FALSE(chat.transcript.enabled);
        CHECK(chat.transcript.folder.empty());
    }

    SECTION("voice sessions do not write transcripts") {
        CHECK_FALSE(AgentRuntime::Call(state.agent, {}, {}).transcript.enabled);
        CHECK_FALSE(Agent::Session::Config{}.transcript.enabled);
    }
}

TEST_CASE("Model request fields combine the always-on object with the selected effort",
          "[agent][compositor][models][extra]") {
    using Json = nlohmann::json;
    Settings::Intelligence::Model model{.provider = "Local", .id = "m"};

    CHECK(AgentProviders::ExtraBody(model).empty());

    model.extraBody = R"({"temperature":0.6,"a":{"x":1}})";
    CHECK(AgentProviders::ExtraBody(model) == model.extraBody);

    model.reasoningEfforts = {"none", "medium", "high"};
    model.effortBodies = R"({"medium":{"a":{"y":2},"temperature":null}})";
    CHECK(Json::parse(AgentProviders::ExtraBody(model)) ==
          Json({{"a", {{"x", 1}, {"y", 2}}}, {"temperature", nullptr}}));

    SECTION("an effort entry can remove a field the always-on object never mentions") {
        auto variant = model;
        variant.effortBodies = R"({"medium":{"reasoning_effort":null,"a":{"y":2}}})";
        CHECK(Json::parse(AgentProviders::ExtraBody(variant)) ==
              Json({{"a", {{"x", 1}, {"y", 2}}}, {"temperature", 0.6}, {"reasoning_effort", nullptr}}));
    }

    model.reasoningEffort = "high";
    CHECK(AgentProviders::ExtraBody(model) == model.extraBody);

    model.reasoningEffort = "medium";
    model.extraBody.clear();
    CHECK(Json::parse(AgentProviders::ExtraBody(model)) == Json({{"a", {{"y", 2}}}, {"temperature", nullptr}}));

    SECTION("malformed text is passed through for the provider to reject") {
        model.effortBodies = "nope";
        CHECK(AgentProviders::ExtraBody(model) == "nope");
        model.effortBodies = R"({"medium":5})";
        CHECK(AgentProviders::ExtraBody(model) == "5");
    }
}

#endif
