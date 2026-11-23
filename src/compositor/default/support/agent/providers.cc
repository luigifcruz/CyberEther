#include "providers.hh"
#include "chatgpt.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include "tools/text.hh"

#include <algorithm>
#include <mutex>
#include <span>
#include <tuple>
#include <utility>

namespace Jetstream {

namespace {

constexpr const char* ProviderKeyService = "CyberEther Intelligence";
constexpr const char* ChatGptHostAccount = "chatgpt-host-id";
constexpr const char* ChatGptRevokeWarning =
    "Signed out locally. Disconnect CyberEther in ChatGPT settings to finish.";

struct Record {
    std::string host;
    std::string value;
};

std::optional<Record> ReadRecord(const std::string& keyId) {
    std::string secret;
    if (keyId.empty() || Platform::ReadSecret(ProviderKeyService, keyId, secret) != Result::SUCCESS) {
        return std::nullopt;
    }
    const auto separator = secret.find('\n');
    if (separator == std::string::npos) {
        return std::nullopt;
    }
    return Record{secret.substr(0, separator), secret.substr(separator + 1)};
}

std::optional<AgentChatGpt::Credentials> ReadCredentials(const std::string& keyId) {
    const auto record = ReadRecord(keyId);
    return record ? AgentChatGpt::Credentials::Decode(record->value) : std::nullopt;
}

std::string ChatGptHostId() {
    std::string hostId;
    if (Platform::ReadSecret(ProviderKeyService, ChatGptHostAccount, hostId) == Result::SUCCESS && !hostId.empty()) {
        return hostId;
    }
    hostId = AgentChatGpt::NewHostId();
    if (!hostId.empty() && Platform::WriteSecret(ProviderKeyService, ChatGptHostAccount, hostId) != Result::SUCCESS) {
        JST_WARN("[AGENT] Can't store the ChatGPT host identifier in the system keychain.");
    }
    return hostId;
}

std::mutex& PlanRefreshMutex() {
    static std::mutex mutex;
    return mutex;
}

void PickOption(std::string& value, std::span<const std::string> options) {
    if (std::find(options.begin(), options.end(), value) == options.end()) {
        value = options.front();
    }
}

bool SupportsVoice(const AgentProviders::Provider& provider) {
    return !Agent::Provider::Profile::Find(AgentProviders::Identify(provider)).livePath.empty();
}

void ResolveModelMetadata(Agent::Provider::Model& model, Agent::Provider::Kind kind,
                          std::optional<Agent::Provider::Catalog>& catalog, const Agent::Cancellation& cancelled) {
    if (model.contextLength != 0 && model.outputLength != 0 && !model.name.empty() &&
        !model.reasoningEfforts.empty()) {
        return;
    }
    if (Agent::Provider::Profile::Find(kind).catalogId.empty() || cancelled()) {
        return;
    }
    if (!catalog) {
        catalog.emplace();
        Agent::Provider::Error error;
        if (catalog->fetch(&error, cancelled) != Result::SUCCESS) {
            JST_WARN("[AGENT] {}", error.message);
        }
    }
    const auto known = catalog->find(kind, model.id);
    if (!known) {
        return;
    }
    if (model.contextLength == 0) {
        model.contextLength = known->contextLength;
    }
    if (model.outputLength == 0) {
        model.outputLength = known->outputLength;
    }
    if (model.name.empty()) {
        model.name = known->name;
    }
    if (model.reasoningEfforts.empty()) {
        model.reasoningEfforts = known->reasoningEfforts;
    }
    if (!model.inputCost || !model.outputCost) {
        model.inputCost = known->inputCost;
        model.outputCost = known->outputCost;
    }
}

void AppendProviderError(std::string& errors, const std::string& provider, const std::string& message) {
    if (!errors.empty()) {
        errors += "\n\n";
    }
    errors += provider + ": " + message;
}

}  // namespace

AgentProviders::AgentProviders(const DefaultCompositorState::AgentState& agent) : agent(agent) {}

std::vector<AgentProviders::Provider> AgentProviders::all() const {
    std::vector<Provider> providers;
    for (const auto& provider : agent.settings.customProviders) {
        if (!provider.name.empty()) {
            providers.push_back(provider);
        }
    }
    return providers;
}

AgentProviders::Provider AgentProviders::find(const std::string& name) const {
    for (const auto& provider : agent.settings.customProviders) {
        if (!provider.name.empty() && provider.name == name) {
            return provider;
        }
    }
    return {};
}

std::vector<AgentProviders::Model> AgentProviders::backends() const {
    std::vector<Model> backends;
    for (const auto& model : agent.settings.models) {
        if (SupportsVoice(find(model.provider))) {
            backends.push_back(model);
        }
    }
    return backends;
}

AgentProviders::VoiceAvailability AgentProviders::availability() const {
    using State = VoiceAvailability::State;

    auto voice = agent.settings.voice;
    PickOption(voice.model, Agent::Live::Models());
    PickOption(voice.voice, Agent::Live::Voices());

    const auto backends = this->backends();
    const bool known = std::any_of(backends.begin(), backends.end(), [&](const auto& model) {
        return Same(model, voice.backend);
    });
    if (!known) {
        voice.backend = backends.empty() ? Model{} : backends.front();
    }

    if (!Agent::Live::Available()) {
        return {State::Unsupported, std::move(voice), "Voice is unavailable in this build."};
    }
    if (voice.backend.id.empty()) {
        return {State::NoBackend, std::move(voice), "Add an OpenAI model in Settings to use voice."};
    }

    const auto backend = Normalize(find(voice.backend.provider));
    if (!Ready(backend, Agent::Provider::Scope::Chat)) {
        return {
            State::MissingKey,
            std::move(voice),
            jst::fmt::format("Add an API key for {} in Settings to use voice.", backend.name),
        };
    }

    return {State::Ready, std::move(voice), ""};
}

Agent::Provider::Kind AgentProviders::Identify(const Provider& provider) {
    if (PlanBilled(provider)) {
        return Agent::Provider::Kind::ChatGPT;
    }
    return Agent::Provider::Profile::Identify(provider.host, provider.chatPath);
}

bool AgentProviders::Valid(const Provider& provider) {
    return !provider.name.empty() && !provider.host.empty();
}

AgentProviders::Provider AgentProviders::Preset(Agent::Provider::Kind kind) {
    const auto& profile = Agent::Provider::Profile::Find(kind);
    const bool custom = kind == Agent::Provider::Kind::Custom;

    return {
        .name = custom ? "" : profile.name,
        .host = profile.host,
        .api = Agent::Provider::Name(profile.api),
        .chatPath = profile.chatPath,
        .modelsPath = profile.modelsPath,
        .auth = kind == Agent::Provider::Kind::ChatGPT ? AgentChatGpt::Auth : "",
        .requestReasoning = profile.requestReasoning,
    };
}

AgentProviders::Provider AgentProviders::Normalize(Provider provider) {
    const auto kind = Identify(provider);
    const auto& defaults = Agent::Provider::Profile::Find(kind);

    if (kind == Agent::Provider::Kind::ChatGPT) {
        provider.host = defaults.host;
        provider.chatPath = defaults.chatPath;
        provider.modelsPath = defaults.modelsPath;
        provider.apiKeyEnv.clear();
    } else {
        provider.account.clear();
    }
    if (provider.api.empty() || kind != Agent::Provider::Kind::Custom) {
        provider.api = Agent::Provider::Name(defaults.api);
    }
    if (provider.chatPath.empty()) {
        provider.chatPath = defaults.chatPath;
    }
    if (provider.modelsPath.empty()) {
        provider.modelsPath = defaults.modelsPath;
    }

    return provider;
}

bool AgentProviders::PlanBilled(const Provider& provider) {
    return provider.auth == AgentChatGpt::Auth;
}

bool AgentProviders::External(const Provider& provider) {
    return !provider.apiKeyEnv.empty();
}

bool AgentProviders::Ready(const Provider& provider, Agent::Provider::Scope scope) {
    const bool hasKey = !provider.apiKeyId.empty() || External(provider);
    return Agent::Provider::Profile::Find(Identify(provider)).ready(scope, hasKey);
}

bool AgentProviders::Transferable(const Provider& stored, const Provider& target) {
    return !stored.apiKeyId.empty() && stored.host == target.host && PlanBilled(stored) == PlanBilled(target);
}

std::string AgentProviders::Effort(const Model& model) {
    const auto& efforts = model.reasoningEfforts;
    if (!model.reasoningEffort.empty() || efforts.empty()) {
        return model.reasoningEffort;
    }

    const auto medium = std::find(efforts.begin(), efforts.end(), "medium");
    if (medium != efforts.end()) {
        return *medium;
    }
    return efforts[efforts.size() / 2];
}

bool AgentProviders::Same(const Model& model, const std::string& provider, const std::string& id) {
    return model.provider == provider && model.id == id;
}

bool AgentProviders::Same(const Model& a, const Model& b) {
    return Same(a, b.provider, b.id);
}

std::shared_ptr<Agent::Provider> AgentProviders::Client(const Provider& provider) {
    auto config = Configure(provider);
    config.apiKey = [provider]() {
        return Key(provider);
    };
    return std::make_shared<Agent::Provider>(std::move(config));
}

DefaultCompositorState::AgentState::ProviderKeyResult AgentProviders::Store(std::string keyId, std::string host,
                                                                            std::string apiKey, bool plan) {
    std::optional<AgentChatGpt::Credentials> replaced;
    {
        std::unique_lock<std::mutex> refreshing;
        if (plan) {
            refreshing = std::unique_lock(PlanRefreshMutex());
            replaced = ReadCredentials(keyId);
        }
        if (Platform::WriteSecret(ProviderKeyService, keyId, host + "\n" + apiKey) != Result::SUCCESS) {
            return {false, "Can't store the API key in the system keychain."};
        }
    }

    const auto stored = plan ? AgentChatGpt::Credentials::Decode(apiKey) : std::nullopt;
    const bool superseded = replaced && (!stored || stored->refreshToken != replaced->refreshToken);
    if (superseded && !AgentChatGpt::Revoke(*replaced)) {
        return {true, "", ChatGptRevokeWarning};
    }
    return {true, ""};
}

DefaultCompositorState::AgentState::ProviderKeyResult AgentProviders::Erase(std::string keyId, bool plan) {
    std::optional<AgentChatGpt::Credentials> credentials;
    {
        std::unique_lock<std::mutex> refreshing;
        if (plan) {
            refreshing = std::unique_lock(PlanRefreshMutex());
            credentials = ReadCredentials(keyId);
        }

        if (Platform::DeleteSecret(ProviderKeyService, keyId) != Result::SUCCESS) {
            return {false, "Can't remove the API key from the system keychain."};
        }
    }

    if (credentials && !AgentChatGpt::Revoke(*credentials)) {
        return {true, "", ChatGptRevokeWarning};
    }
    return {true, ""};
}

DefaultCompositorState::AgentState::ModelListResult AgentProviders::Models(std::vector<Provider> providers,
                                                                           Agent::Cancellation cancelled) {
    DefaultCompositorState::AgentState::ModelListResult result;
    std::optional<Agent::Provider::Catalog> catalog;
    for (const auto& provider : providers) {
        if (cancelled()) {
            break;
        }

        std::string keyError;
        if (External(provider) && Key(provider, &keyError).empty()) {
            result.health[provider.name] = Agent::Provider::Error::Kind::MissingCredentials;
            AppendProviderError(result.error, provider.name, keyError);
            continue;
        }

        std::vector<Agent::Provider::Model> models;
        Agent::Provider::Error error;
        const bool listed = Client(provider)->models(models, &error, cancelled) == Result::SUCCESS;
        result.health[provider.name] = listed ? Agent::Provider::Error::Kind::None : error.kind;
        if (!listed) {
            AppendProviderError(result.error, provider.name, error.message);
            continue;
        }
        const auto kind = Identify(provider);
        result.models.reserve(result.models.size() + models.size());
        for (auto& model : models) {
            ResolveModelMetadata(model, kind, catalog, cancelled);
            if (kind == Agent::Provider::Kind::ChatGPT) {
                model.inputCost.reset();
                model.outputCost.reset();
            }
            result.models.push_back({
                .provider = provider.name,
                .id = std::move(model.id),
                .name = std::move(model.name),
                .contextLength = model.contextLength,
                .outputLength = model.outputLength,
                .reasoningEfforts = std::move(model.reasoningEfforts),
                .priced = model.inputCost.has_value() && model.outputCost.has_value(),
                .inputCost = model.inputCost.value_or(0.0),
                .outputCost = model.outputCost.value_or(0.0),
            });
        }
    }
    std::sort(result.models.begin(), result.models.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.provider, lhs.id) < std::tie(rhs.provider, rhs.id);
    });
    return result;
}

std::pair<bool, std::string> AgentProviders::Test(Provider provider, std::string apiKey,
                                                 std::optional<Provider> stored, Agent::Cancellation cancelled) {
    if (cancelled()) {
        return {false, "Connection test cancelled."};
    }

    std::string keyError;
    auto key = std::move(apiKey);
    if (!key.empty() && PlanBilled(provider)) {
        const auto credentials = AgentChatGpt::Credentials::Decode(key);
        key = credentials ? credentials->accessToken : std::string{};
    }
    if (key.empty() && External(provider)) {
        key = Key(provider, &keyError);
    } else if (key.empty() && stored) {
        stored->apiKeyEnv.clear();
        key = Key(*stored);
    }
    if (!keyError.empty()) {
        return {false, keyError};
    }
    const bool hasKey = !key.empty();

    auto config = Configure(provider);
    config.apiKey = [key = std::move(key)]() {
        return key;
    };

    std::vector<Agent::Provider::Model> models;
    Agent::Provider::Error error;
    const Agent::Provider client(std::move(config));
    if (client.models(models, &error, cancelled) != Result::SUCCESS) {
        return {false, error.message};
    }

    std::string message = jst::fmt::format("Connected. Found {} model{}.",
                                           models.size(),
                                           models.size() == 1 ? "" : "s");
    if (!client.profile().ready(Agent::Provider::Scope::Chat, hasKey)) {
        message += " The model catalog is public. Add an API key before starting a chat.";
    }
    return {true, message};
}

bool AgentProviders::SignInAvailable() {
    static const bool available = AgentChatGpt::Available() && Platform::SecretStoreAvailable();
    return available;
}

DefaultCompositorState::AgentState::SignInResult AgentProviders::SignIn(
    std::shared_ptr<DefaultCompositorState::AgentState::SignInLink> link, Agent::Cancellation cancelled) {
    const auto open = [link](const std::string& url) {
        std::lock_guard lock(link->mutex);
        link->url = url;
    };
    auto signedIn = AgentChatGpt::SignIn(ChatGptHostId(), open, cancelled);
    if (!signedIn.success || !signedIn.credentials) {
        return {.success = false, .message = std::move(signedIn.message)};
    }

    auto record = signedIn.credentials->encode();
    auto email = signedIn.credentials->email;
    const auto [listed, listing] = Test(Preset(Agent::Provider::Kind::ChatGPT), record, std::nullopt, cancelled);
    if (!listed) {
        AgentChatGpt::Revoke(*signedIn.credentials);
        return {.success = false, .message = listing};
    }

    const auto account = email.empty() ? std::string("ChatGPT") : email;
    return {
        .success = true,
        .message = jst::fmt::format("Signed in as {}. {}", account, listing),
        .record = std::move(record),
        .email = std::move(email),
    };
}

std::string AgentProviders::Key(const Provider& provider, std::string* error) {
    const auto missing = [&](std::string message) {
        JST_WARN("[AGENT] {}", message);
        if (error) {
            *error = std::move(message);
        }
        return std::string{};
    };

    if (!provider.apiKeyEnv.empty()) {
        std::string value;
        if (Platform::EnvironmentVariable(provider.apiKeyEnv, value) != Result::SUCCESS || Text::IsBlank(value)) {
            return missing(jst::fmt::format("Provider '{}' reads its API key from the environment variable {}, "
                                            "which is not set.",
                                            provider.name,
                                            provider.apiKeyEnv));
        }
        return Text::TrimBlank(value);
    }

    std::string secret;
    if (provider.apiKeyId.empty()) {
        return {};
    }

    if (Platform::ReadSecret(ProviderKeyService, provider.apiKeyId, secret) != Result::SUCCESS) {
        JST_WARN("[AGENT] Can't read the stored API key for provider '{}'.", provider.name);
        return {};
    }

    const auto separator = secret.find('\n');
    const bool hostMatches = separator != std::string::npos && secret.compare(0, separator, provider.host) == 0;
    if (!hostMatches) {
        return {};
    }

    if (PlanBilled(provider)) {
        return PlanToken(provider, secret.substr(separator + 1));
    }
    if (AgentChatGpt::Credentials::Decode(secret.substr(separator + 1))) {
        return {};
    }
    return secret.substr(separator + 1);
}

std::string AgentProviders::PlanToken(const Provider& provider, const std::string& record) {
    auto credentials = AgentChatGpt::Credentials::Decode(record);
    if (!credentials) {
        JST_WARN("[AGENT] The ChatGPT sign-in for provider '{}' is unreadable. Sign in again in Settings.",
                 provider.name);
        return {};
    }
    if (!credentials->expiring(AgentChatGpt::Now())) {
        return credentials->accessToken;
    }

    std::lock_guard lock(PlanRefreshMutex());
    const auto latest = ReadRecord(provider.apiKeyId);
    credentials = latest && latest->host == provider.host ? AgentChatGpt::Credentials::Decode(latest->value)
                                                          : std::nullopt;
    if (!credentials) {
        return {};
    }
    if (!credentials->expiring(AgentChatGpt::Now())) {
        return credentials->accessToken;
    }

    const auto refreshed = AgentChatGpt::Refresh(*credentials);
    if (!refreshed.success) {
        JST_WARN("[AGENT] {}{}", refreshed.message, refreshed.terminal ? " Sign in to ChatGPT again in Settings." : "");
        return credentials->expiresAt > AgentChatGpt::Now() ? credentials->accessToken : std::string{};
    }

    const auto value = provider.host + "\n" + credentials->encode();
    if (Platform::WriteSecret(ProviderKeyService, provider.apiKeyId, value) != Result::SUCCESS) {
        JST_WARN("[AGENT] Can't store the refreshed ChatGPT sign-in in the system keychain.");
    }
    return credentials->accessToken;
}

Agent::Provider::Config AgentProviders::Configure(const Provider& value) {
    const auto provider = Normalize(value);

    return {
        .name = provider.name,
        .kind = Identify(provider),
        .host = provider.host,
        .api = Agent::Provider::Parse(provider.api, Agent::Provider::Api::ChatCompletions),
        .path = provider.chatPath,
        .modelsPath = provider.modelsPath,
        .requireApiKey = External(provider),
        .timeoutSeconds = 300,
        .retries = 5,
        .requestReasoning = provider.requestReasoning,
    };
}

}  // namespace Jetstream
