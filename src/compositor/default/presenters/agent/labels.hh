#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_LABELS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_LABELS_HH

#include "../../support/agent/providers.hh"
#include "../../model/state.hh"

#include "jetstream/logger.hh"
#include "jetstream/settings.hh"

#include <cctype>
#include <cmath>
#include <string>

namespace Jetstream {

struct AgentProviderStatus {
    std::string label;
    bool ready = false;
    bool pending = false;
};

class AgentLabels {
 public:
    static std::string Model(const Settings::Intelligence::Model& model);
    static std::string Title(const std::string& id);
    static std::string Effort(const std::string& effort);
    static std::string Tokens(const U64 tokens);
    static std::string Price(F64 dollars);
    static AgentProviderStatus Health(Agent::Provider::Error::Kind kind);
    static AgentProviderStatus Status(const DefaultCompositorState::AgentState& agent,
                                      const Settings::Intelligence::Provider& value);
    static std::string Voice(AgentProviders::VoiceAvailability::State state);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_AGENT_LABELS_HH
