#include "providers.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include <algorithm>
#include <span>
#include <tuple>
#include <utility>

namespace Jetstream {

namespace {

constexpr const char* ProviderKeyService = "CyberEther Intelligence";

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
        .requestReasoning = profile.requestReasoning,
    };
}

AgentProviders::Provider AgentProviders::Normalize(Provider provider) {
    const auto& defaults = Agent::Provider::Profile::Find(Identify(provider));

    if (provider.api.empty()) {
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

bool AgentProviders::Ready(const Provider& provider, Agent::Provider::Scope scope) {
    return Agent::Provider::Profile::Find(Identify(provider)).ready(scope, !provider.apiKeyId.empty());
}

bool AgentProviders::Transferable(const Provider& stored, const Provider& target) {
    return !stored.apiKeyId.empty() && stored.host == target.host;
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
                                                                            std::string apiKey) {
    if (Platform::WriteSecret(ProviderKeyService, keyId, host + "\n" + apiKey) != Result::SUCCESS) {
        return {false, "Can't store the API key in the system keychain."};
    }

    return {true, ""};
}

DefaultCompositorState::AgentState::ProviderKeyResult AgentProviders::Erase(std::string keyId) {
    if (Platform::DeleteSecret(ProviderKeyService, keyId) != Result::SUCCESS) {
        return {false, "Can't remove the API key from the system keychain."};
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
            result.models.push_back({
                .provider = provider.name,
                .id = std::move(model.id),
                .name = std::move(model.name),
                .contextLength = model.contextLength,
                .outputLength = model.outputLength,
                .reasoningEfforts = std::move(model.reasoningEfforts),
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

    auto key = apiKey.empty() && stored ? Key(*stored) : std::move(apiKey);
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

std::string AgentProviders::Key(const Provider& provider) {
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

    return secret.substr(separator + 1);
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
        .timeoutSeconds = 300,
        .retries = 5,
        .requestReasoning = provider.requestReasoning,
    };
}

}  // namespace Jetstream
