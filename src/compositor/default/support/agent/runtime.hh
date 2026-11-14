#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_RUNTIME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_RUNTIME_HH

#include "executor.hh"
#include "mcp.hh"
#include "../../model/state.hh"

#include "jetstream/agent/session.hh"
#include "jetstream/settings.hh"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct DefaultCompositorState;
struct DefaultCompositorCallbacks;

class AgentRuntime {
 public:
    struct Voice {
        Agent::Session::LiveFactory live;
        Agent::Session::AudioFactory audio;
    };

    AgentRuntime(DefaultCompositorState& state, const DefaultCompositorCallbacks& callbacks, Voice voice = {});
    ~AgentRuntime();

    AgentRuntime(const AgentRuntime&) = delete;
    AgentRuntime& operator=(const AgentRuntime&) = delete;

    void shutdown();
    void poll();

    bool submit(const std::string& text);
    bool listen(const Settings::Intelligence::Voice& voice);
    void hangup();
    void setMuted(bool value);
    Agent::Session& session();
    AgentExecutor& executor();

    static Agent::Session::Config Chat(const DefaultCompositorState::AgentState& agent,
                                       Agent::Session::ToolExecutor executeTool);
    static Agent::Session::Config Call(const DefaultCompositorState::AgentState& agent,
                                       const Settings::Intelligence::Voice& voice,
                                       Agent::Session::ToolExecutor executeTool);
    static std::string SystemPrompt();
    static std::string McpPrompt();
    static std::string VoicePrompt();

 private:
    struct Mute;

    DefaultCompositorState& state;
    const DefaultCompositorCallbacks& callbacks;
    std::shared_ptr<Mute> mute;
    Agent::Session conversation;
    AgentExecutor tools;
    Agent::Session::Cursor cursor;
    AgentMcp mcp;
    std::optional<std::pair<std::string, U64>> served;

    void publish();
    void serve();

    static Agent::Session::AudioFactory MuteAudio(Agent::Session::AudioFactory factory, std::shared_ptr<Mute> mute);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_RUNTIME_HH
