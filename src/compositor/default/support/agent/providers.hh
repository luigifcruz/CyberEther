#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_PROVIDERS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_PROVIDERS_HH

#include "../../model/state.hh"

#include "jetstream/agent/provider.hh"
#include "jetstream/settings.hh"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

class AgentProviders {
 public:
    using Provider = Settings::Intelligence::Provider;
    using Model = Settings::Intelligence::Model;
    using Voice = Settings::Intelligence::Voice;

    struct VoiceAvailability {
        enum class State {
            Unsupported,
            NoBackend,
            MissingKey,
            Ready,
        };

        State state = State::Unsupported;
        Voice voice;
        std::string message;
    };

    explicit AgentProviders(const DefaultCompositorState::AgentState& agent);

    std::vector<Provider> all() const;
    Provider find(const std::string& name) const;
    std::vector<Model> backends() const;
    VoiceAvailability availability() const;

    static Agent::Provider::Kind Identify(const Provider& provider);
    static bool Valid(const Provider& provider);
    static Provider Preset(Agent::Provider::Kind kind);
    static Provider Normalize(Provider provider);
    static bool Ready(const Provider& provider, Agent::Provider::Scope scope);
    static bool Transferable(const Provider& stored, const Provider& target);
    static std::string Effort(const Model& model);
    static bool Same(const Model& model, const std::string& provider, const std::string& id);
    static bool Same(const Model& a, const Model& b);
    static std::shared_ptr<Agent::Provider> Client(const Provider& provider);

    static DefaultCompositorState::AgentState::ProviderKeyResult Store(std::string keyId, std::string host,
                                                                       std::string apiKey);
    static DefaultCompositorState::AgentState::ProviderKeyResult Erase(std::string keyId);
    static DefaultCompositorState::AgentState::ModelListResult Models(std::vector<Provider> providers,
                                                                      Agent::Cancellation cancelled);
    static std::pair<bool, std::string> Test(Provider provider, std::string apiKey, std::optional<Provider> stored,
                                             Agent::Cancellation cancelled);

 private:
    static std::string Key(const Provider& provider);
    static Agent::Provider::Config Configure(const Provider& provider);

    const DefaultCompositorState::AgentState& agent;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_PROVIDERS_HH
