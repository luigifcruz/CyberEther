#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "compositor/default/actions/agent.hh"
#include "compositor/default/support/agent/chatgpt.hh"
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

    SECTION("the Anthropic endpoint uses the Messages format") {
        const Settings::Intelligence::Provider stored{
            .name = "Anthropic",
            .host = "https://api.anthropic.com",
            .api = "chat_completions",
            .chatPath = "/v1/messages",
        };
        REQUIRE(AgentProviders::Identify(stored) == Kind::Anthropic);

        const auto normalized = AgentProviders::Normalize(stored);
        CHECK(normalized.api == "messages");
        CHECK(normalized.modelsPath == "/v1/models");
        CHECK(AgentProviders::Client(normalized)->config().api == Agent::Provider::Api::Messages);
        CHECK(AgentProviders::Preset(Kind::Anthropic).api == "messages");
    }

    SECTION("a custom endpoint can use the Messages format") {
        const Settings::Intelligence::Provider stored{
            .name = "Proxy",
            .host = "http://localhost:8080",
            .api = "messages",
            .chatPath = "/v1/messages",
        };
        REQUIRE(AgentProviders::Identify(stored) == Kind::Custom);

        CHECK(AgentProviders::Normalize(stored).api == "messages");
        CHECK(AgentProviders::Client(stored)->config().api == Agent::Provider::Api::Messages);
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

TEST_CASE("ChatGPT providers are identified by their sign-in, not their endpoint",
          "[agent][compositor][providers][chatgpt]") {
    using Kind = Agent::Provider::Kind;

    const auto preset = AgentProviders::Preset(Kind::ChatGPT);
    CHECK(preset.name == "ChatGPT Plan");
    CHECK(preset.auth == AgentChatGpt::Auth);
    CHECK(AgentProviders::Identify(preset) == Kind::ChatGPT);

    auto edited = preset;
    edited.host = "https://example.invalid";
    edited.chatPath = "/chat";
    edited.apiKeyEnv = "OPENAI_API_KEY";
    const auto normalized = AgentProviders::Normalize(edited);
    CHECK(normalized.host == "https://api.openai.com");
    CHECK(normalized.chatPath == "/v1/responses");
    CHECK(normalized.api == "responses");
    CHECK(normalized.apiKeyEnv.empty());
    CHECK(AgentProviders::Client(normalized)->config().kind == Kind::ChatGPT);

    auto account = preset;
    account.account = "person@example.com";
    CHECK(AgentProviders::Normalize(account).account == "person@example.com");
    account.auth.clear();
    CHECK(AgentProviders::Normalize(account).account.empty());

    auto openai = AgentProviders::Preset(Kind::OpenAI);
    REQUIRE(AgentProviders::Identify(openai) == Kind::OpenAI);
    openai.apiKeyId = "openai-key";
    auto plan = preset;
    plan.apiKeyId = "plan-key";
    CHECK_FALSE(AgentProviders::Transferable(openai, preset));
    CHECK_FALSE(AgentProviders::Transferable(plan, AgentProviders::Preset(Kind::OpenAI)));
    CHECK(AgentProviders::Transferable(plan, preset));
}

TEST_CASE("ChatGPT credentials round-trip and refresh before expiry", "[agent][compositor][providers][chatgpt]") {
    const AgentChatGpt::Credentials credentials{
        .clientId = "oaiapp_test",
        .accessToken = "access",
        .refreshToken = "refresh",
        .scope = "openid offline_access chatgpt.tokens.use.direct",
        .subject = "user-1",
        .email = "user@example.invalid",
        .expiresAt = 10'000,
    };

    const auto decoded = AgentChatGpt::Credentials::Decode(credentials.encode());
    REQUIRE(decoded.has_value());
    CHECK(decoded->clientId == credentials.clientId);
    CHECK(decoded->accessToken == credentials.accessToken);
    CHECK(decoded->refreshToken == credentials.refreshToken);
    CHECK(decoded->subject == credentials.subject);
    CHECK(decoded->email == credentials.email);
    CHECK(decoded->expiresAt == credentials.expiresAt);
    CHECK(decoded->planEnabled());

    CHECK_FALSE(credentials.expiring(9'000));
    CHECK(credentials.expiring(9'900));
    CHECK(credentials.expiring(10'000));

    auto withoutPlan = credentials;
    withoutPlan.scope = "openid chatgpt.tokens.use";
    CHECK_FALSE(withoutPlan.planEnabled());

    CHECK_FALSE(AgentChatGpt::Credentials::Decode("sk-plain-api-key").has_value());
    CHECK_FALSE(AgentChatGpt::Credentials::Decode(R"({"client_id":"oaiapp_test"})").has_value());
}

TEST_CASE("ChatGPT authorization requests register once and reuse the issued client",
          "[agent][compositor][providers][chatgpt]") {
    AgentChatGpt::Authorization authorization{
        .hostId = "urn:uuid:00000000-0000-4000-8000-000000000000",
        .redirectUri = "http://127.0.0.1:1455/auth/callback",
        .state = "state-1",
        .nonce = "nonce-1",
        .challenge = "challenge-1",
    };

    const auto registration = AgentChatGpt::AuthorizeUrl(authorization);
    REQUIRE(registration.starts_with("https://auth.openai.com/api/accounts/authorize?"));
    const auto first = AgentChatGpt::ParseQuery(registration.substr(registration.find('?') + 1));
    CHECK(first.at("client_id") == "dynamic_agent_client");
    CHECK(first.at("agent_name_hint") == "CyberEther");
    CHECK(first.at("ext_agent_host_id") == authorization.hostId);
    CHECK(first.at("redirect_uri") == authorization.redirectUri);
    CHECK(first.at("scope") == "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct");
    CHECK(first.at("resource") == "https://api.openai.com/v1");
    CHECK(first.at("response_type") == "code");
    CHECK(first.at("code_challenge_method") == "S256");
    CHECK(first.at("state") == authorization.state);
    CHECK(first.at("nonce") == authorization.nonce);
    CHECK_FALSE(first.contains("id_token_hint"));
    CHECK_FALSE(first.contains("login_hint"));

    const auto callback = AgentChatGpt::ParseQuery("code=a%2Fb%3D&state=s+1&client_id=oaiapp_x&empty&code=ignored");
    CHECK(callback.at("code") == "a/b=");
    CHECK(callback.at("state") == "s 1");
    CHECK(callback.at("client_id") == "oaiapp_x");
    CHECK(callback.at("empty").empty());
}

TEST_CASE("ChatGPT PKCE challenges are unpadded base64url SHA-256 digests", "[agent][compositor][providers][chatgpt]") {
    if (!AgentChatGpt::Available()) {
        SKIP("Sign in with ChatGPT is unavailable in this build.");
    }
    CHECK(AgentChatGpt::Challenge("dBjftJeZ4CVP-mJ92K9OSRVHVDKiTc6iSJYhvTm4r8k") ==
          "eaeoO5ynwr6rfFEOKTLk-6T9WIOvp7BWGM-YMWVtLLc");

    const auto hostId = AgentChatGpt::NewHostId();
    REQUIRE(hostId.size() == std::string("urn:uuid:").size() + 36);
    CHECK(hostId.starts_with("urn:uuid:"));
    CHECK(hostId[9 + 14] == '4');
    CHECK(hostId != AgentChatGpt::NewHostId());
}

TEST_CASE("A finished ChatGPT sign-in passes the connection test and saves its draft only",
          "[agent][compositor][providers][chatgpt]") {
    const bool sameSession = GENERATE(true, false);
    CAPTURE(sameSession);
    SettingsScope scope;

    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> messages;
    callbacks.notify = [](auto, auto, const std::string&) {};
    callbacks.enqueueMail = [&](Mail&& mail) { messages.push_back(std::move(mail)); };
    AgentRuntime runtime(state, callbacks);
    AgentActions actions(state, callbacks, runtime);

    auto& providers = state.agent.providers;
    providers.editSession = 4;
    auto& signIn = providers.signIn;
    std::promise<DefaultCompositorState::AgentState::SignInResult> result;
    signIn.running = true;
    signIn.session = sameSession ? 4 : 3;
    signIn.revision = 7;
    signIn.draft = AgentProviders::Preset(Agent::Provider::Kind::ChatGPT);
    signIn.cancel = std::make_shared<std::atomic<bool>>(false);
    signIn.future = result.get_future();
    result.set_value({
        .success = true,
        .message = "Signed in as user@example.invalid. Connected. Found 2 models.",
        .record = "record",
        .email = "user@example.invalid",
    });

    actions.reconcile();

    CHECK_FALSE(signIn.running);
    const auto& test = state.agent.providerTest;
    CHECK(test.completed == sameSession);
    CHECK(test.success == sameSession);
    CHECK(signIn.record == (sameSession ? "record" : ""));
    if (sameSession) {
        CHECK(test.revision == 7);
        CHECK(signIn.email == "user@example.invalid");
    }

    std::vector<MailSaveAgentProvider> saves;
    for (auto& message : messages) {
        if (auto* save = std::get_if<MailSaveAgentProvider>(&message)) {
            saves.push_back(*save);
        }
    }
    REQUIRE(saves.size() == (sameSession ? 1u : 0u));
    if (sameSession) {
        CHECK(saves.front().value.name == "ChatGPT Plan");
        CHECK(saves.front().value.auth == AgentChatGpt::Auth);
        CHECK_FALSE(saves.front().replaceName.has_value());
    }
}

TEST_CASE("Removing a provider drops its pending ChatGPT sign-in", "[agent][compositor][providers][chatgpt]") {
    SettingsScope scope;

    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> messages;
    callbacks.notify = [](auto, auto, const std::string&) {};
    callbacks.enqueueMail = [&](Mail&& mail) { messages.push_back(std::move(mail)); };
    AgentRuntime runtime(state, callbacks);
    AgentActions actions(state, callbacks, runtime);

    auto stored = AgentProviders::Preset(Agent::Provider::Kind::ChatGPT);
    state.agent.settings.customProviders.push_back(stored);

    auto& providers = state.agent.providers;
    providers.editSession = 3;
    auto& signIn = providers.signIn;
    std::promise<DefaultCompositorState::AgentState::SignInResult> result;
    signIn.running = true;
    signIn.session = 3;
    signIn.draft = stored;
    signIn.replaceName = stored.name;
    signIn.cancel = std::make_shared<std::atomic<bool>>(false);
    signIn.future = result.get_future();

    REQUIRE(actions.handle(MailRemoveAgentProvider{.name = stored.name}) == Result::SUCCESS);
    CHECK(signIn.cancel->load());
    CHECK_FALSE(signIn.draft.has_value());

    result.set_value({.success = true, .message = "Signed in.", .record = "record", .email = "user@example.invalid"});
    actions.reconcile();
    for (const auto& message : messages) {
        CHECK_FALSE(std::holds_alternative<MailSaveAgentProvider>(message));
    }
    CHECK(AgentProviders(state.agent).find(stored.name).name.empty());
}

TEST_CASE("A finished ChatGPT sign-in saves under a free name instead of overwriting another provider",
          "[agent][compositor][providers][chatgpt]") {
    SettingsScope scope;

    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> messages;
    callbacks.notify = [](auto, auto, const std::string&) {};
    callbacks.enqueueMail = [&](Mail&& mail) { messages.push_back(std::move(mail)); };
    AgentRuntime runtime(state, callbacks);
    AgentActions actions(state, callbacks, runtime);

    auto existing = AgentProviders::Preset(Agent::Provider::Kind::ChatGPT);
    existing.apiKeyId = "other";
    state.agent.settings.customProviders.push_back(existing);

    auto& providers = state.agent.providers;
    providers.editSession = 2;
    auto& signIn = providers.signIn;
    std::promise<DefaultCompositorState::AgentState::SignInResult> result;
    signIn.running = true;
    signIn.session = 2;
    signIn.draft = AgentProviders::Preset(Agent::Provider::Kind::ChatGPT);
    signIn.cancel = std::make_shared<std::atomic<bool>>(false);
    signIn.future = result.get_future();
    result.set_value({.success = true, .message = "Signed in.", .record = "record", .email = "user@example.invalid"});

    actions.reconcile();

    CHECK(signIn.record == "record");
    std::vector<MailSaveAgentProvider> saves;
    for (auto& message : messages) {
        if (auto* save = std::get_if<MailSaveAgentProvider>(&message)) {
            saves.push_back(*save);
        }
    }
    REQUIRE(saves.size() == 1);
    CHECK(saves.front().value.name == "ChatGPT Plan 2");
    CHECK_FALSE(saves.front().replaceName.has_value());
}

#endif
