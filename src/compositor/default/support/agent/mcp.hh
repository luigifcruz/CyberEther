#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_MCP_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_MCP_HH

#include "jetstream/agent/provider.hh"
#include "jetstream/agent/session.hh"
#include "jetstream/types.hh"

#include <memory>
#include <string>
#include <vector>

namespace Jetstream {

class AgentMcp {
 public:
    struct Config {
        std::string address;
        U16 port = 0;
        std::string instructions;
        std::vector<Agent::Provider::Function> tools;
        Agent::Session::ToolExecutor executeTool;
    };

    AgentMcp();
    ~AgentMcp();

    AgentMcp(const AgentMcp&) = delete;
    AgentMcp& operator=(const AgentMcp&) = delete;

    Result start(Config config);
    void stop();

    bool running() const;
    const std::string& url() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_MCP_HH
