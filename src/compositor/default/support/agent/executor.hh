#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_EXECUTOR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_EXECUTOR_HH

#include "jetstream/agent/session.hh"
#include "jetstream/agent/tools.hh"
#include "jetstream/types.hh"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace Jetstream {

struct DefaultCompositorState;
struct DefaultCompositorCallbacks;
class Flowgraph;

class AgentExecutor {
 public:
    using Layouts = std::unordered_map<std::string, Agent::Tools::Layout>;

    AgentExecutor(DefaultCompositorState& state, const DefaultCompositorCallbacks& callbacks);
    ~AgentExecutor();

    Agent::Session::ToolExecutor bind();
    Agent::Session::ToolExecutor focused();
    void layout(const std::string& flowgraph, Agent::Tools::Layout layout);
    void poll();
    void shutdown();

 private:
    struct Shared;
    struct Binding;
    std::shared_ptr<Shared> shared;
    DefaultCompositorState& state;
    const DefaultCompositorCallbacks& callbacks;
    Layouts layouts;

    void release(Layouts layouts);
    void dispatch();
    Agent::Tools::Output execute(const std::string& name, const std::string& args,
                                 const Agent::Cancellation& shouldAbort,
                                 std::shared_ptr<Binding> binding);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_EXECUTOR_HH
