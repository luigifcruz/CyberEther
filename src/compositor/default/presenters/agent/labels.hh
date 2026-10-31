#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_LABELS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_LABELS_HH

#include "../../support/agent/providers.hh"
#include "../../model/state.hh"

#include "jetstream/logger.hh"
#include "jetstream/settings.hh"

#include <cctype>
#include <string>

namespace Jetstream {

struct AgentProviderStatus {
    std::string label;
    bool ready = false;
    bool pending = false;
};

class AgentLabels {
 public:
    static std::string Model(const Settings::Intelligence::Model& model) {
        if (!model.name.empty()) {
            return model.name;
        }
        const auto slash = model.id.rfind('/');
        return slash == std::string::npos ? model.id : model.id.substr(slash + 1);
    }

    static std::string Effort(const std::string& effort) {
        if (effort == "xhigh") {
            return "Extra High";
        }
        std::string label = effort;
        if (!label.empty()) {
            label.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(label.front())));
        }
        return label;
    }

    static std::string Tokens(const U64 tokens) {
        if (tokens < 1000) {
            return jst::fmt::format("{}", tokens);
        }
        if (tokens < 1000000) {
            return jst::fmt::format("{}k", (tokens + 500) / 1000);
        }
        auto value = jst::fmt::format("{:.2f}", static_cast<F64>(tokens) / 1e6);
        value.erase(value.find_last_not_of('0') + 1);
        if (value.back() == '.') {
            value.pop_back();
        }
        return value + "M";
    }

    static std::string Context(U64 tokens) {
        return tokens == 0 ? "" : Tokens(tokens);
    }

    static AgentProviderStatus Health(Agent::Provider::Error::Kind kind) {
        using Kind = Agent::Provider::Error::Kind;
        switch (kind) {
            case Kind::None:
                return {.label = "Ready", .ready = true};
            case Kind::MissingCredentials:
                return {.label = "Key not set"};
            case Kind::Authentication:
                return {.label = "Invalid key"};
            case Kind::Permission:
                return {.label = "No access"};
            case Kind::NotFound:
                return {.label = "Wrong path"};
            case Kind::RateLimit:
                return {.label = "Rate limited"};
            case Kind::Quota:
                return {.label = "Out of quota"};
            case Kind::Timeout:
                return {.label = "Timed out"};
            case Kind::Connection:
                return {.label = "Unreachable"};
            case Kind::Tls:
                return {.label = "TLS error"};
            case Kind::Server:
                return {.label = "Server error"};
            default:
                return {.label = "Error"};
        }
    }

    static AgentProviderStatus Status(const DefaultCompositorState::AgentState& agent,
                                      const Settings::Intelligence::Provider& value) {
        const auto provider = AgentProviders::Normalize(value);
        if (!AgentProviders::Ready(provider, Agent::Provider::Scope::Chat)) {
            return {.label = "Key not set"};
        }
        const auto& health = agent.providers.health;
        const auto entry = health.find(provider.name);
        if (entry == health.end()) {
            return {.label = "Checking...", .pending = true};
        }
        return Health(entry->second);
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_LABELS_HH
