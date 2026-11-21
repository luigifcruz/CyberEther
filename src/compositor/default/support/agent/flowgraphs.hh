#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_FLOWGRAPHS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_FLOWGRAPHS_HH

#include "jetstream/agent/tools.hh"
#include "jetstream/render/sakura/callback.hh"

#include <memory>
#include <string>
#include <vector>

namespace Jetstream {

struct DefaultCompositorState;
struct DefaultCompositorCallbacks;
class Flowgraph;

class AgentFlowgraphs {
 public:
    using Completion = Sakura::Callback<void(Agent::Tools::Output, std::shared_ptr<Flowgraph>)>;

    AgentFlowgraphs(DefaultCompositorState& state, const DefaultCompositorCallbacks& callbacks);

    static bool Handles(const std::string& name);
    static std::vector<Agent::Tools::Definition> Definitions();

    void execute(const std::shared_ptr<Flowgraph>& flowgraph,
                 const std::string& name,
                 const std::string& args,
                 Completion done) const;

 private:
    DefaultCompositorState& state;
    const DefaultCompositorCallbacks& callbacks;

    Agent::Tools::Output list() const;
    void open(const std::string& args, Completion done) const;
    Agent::Tools::Output listOpen(const std::shared_ptr<Flowgraph>& flowgraph) const;
    void focus(const std::shared_ptr<Flowgraph>& flowgraph, const std::string& args, Completion done) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_FLOWGRAPHS_HH
