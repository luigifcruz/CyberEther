#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_TILES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_TILES_HH

#include "jetstream/agent/provider.hh"
#include "jetstream/agent/tools.hh"

#include <memory>
#include <string>
#include <vector>

namespace Jetstream {

struct DefaultCompositorState;
struct DefaultCompositorCallbacks;
class Flowgraph;

class AgentTiles {
 public:
    AgentTiles(DefaultCompositorState& state, const DefaultCompositorCallbacks& callbacks);

    static bool Handles(const std::string& name);
    static std::vector<Agent::Provider::Function> Functions();

    Agent::Tools::Output execute(const std::shared_ptr<Flowgraph>& flowgraph,
                                 const std::string& name,
                                 const std::string& args) const;

 private:
    struct Tile;

    DefaultCompositorState& state;
    const DefaultCompositorCallbacks& callbacks;

    static std::vector<Tile> Catalog(Flowgraph& flowgraph);
    Agent::Tools::Output inspect(const std::string& flowgraphId, Flowgraph& flowgraph) const;
    Agent::Tools::Output arrange(const std::string& flowgraphId, Flowgraph& flowgraph, const std::string& args) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_TILES_HH
