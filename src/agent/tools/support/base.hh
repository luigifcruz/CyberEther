#ifndef JETSTREAM_AGENT_TOOLS_SUPPORT_BASE_HH
#define JETSTREAM_AGENT_TOOLS_SUPPORT_BASE_HH

#include "jetstream/agent/tools.hh"
#include "jetstream/config.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/parser.hh"
#include "jetstream/registry.hh"
#include "tools/json.hh"
#include "tools/value.hh"

#include <nlohmann/json.hpp>

#include <functional>
#include <optional>

namespace Jetstream::Agent::Detail {

using Json = nlohmann::json;
using Graph = std::shared_ptr<Flowgraph>;
using Host = Tools::Host;
using Layout = Tools::Layout;
using Definition = Tools::Definition;
using Call = Tools::Call;

struct Invocation {
    Host& host;
    const Graph& flowgraph;
    const std::string& created;
    const Json& args;
};

struct Entry {
    struct NewGraph {
        std::string name;
        std::string error;
    };

    Definition definition;
    std::function<Tools::Output(const Invocation&)> execute;
    NewGraph (*newGraph)(const Json& args) = nullptr;
    void (*annotate)(Json& body, const Json& args, const Layout& layout) = nullptr;

    static Entry FromJson(Definition definition, Json (*execute)(const Graph&, const Json&));
    static Entry FromJson(Definition definition, Json (*execute)(const Invocation&));
};

class Toolbox {
 public:
    static const std::vector<Entry>& All();
    static const Entry* Find(const std::string& name);

    static std::vector<Entry> Catalog();
    static std::vector<Entry> Inspect();
    static std::vector<Entry> Patch();
    static std::vector<Entry> Connections();
    static std::vector<Entry> Environment();
    static std::vector<Entry> Capture();
    static std::vector<Entry> Text();
};

class Response {
 public:
    static Json Error(const std::string& message);
    static bool Failed(const Json& body);
    static Tools::Output Reply(const Json& body, std::vector<std::string> images = {});
    static Tools::Output Failure(const std::string& message);
};

class Arguments {
 public:
    static U64 Clamp(const Json& args, const std::string& key, I64 fallback, I64 low, I64 high);
};

class BlockText {
 public:
    struct Patch {
        std::string text;
        std::string error;
        Json changes = Json::array();
        U64 lineCount = 0;
    };

    static std::string Read(const Graph& flowgraph, const std::string& name, const std::string& field,
                            std::string& text);
    static Patch Apply(const std::string& text, const std::string& patch);
    static Json Range(const std::string& text, const Json& args, std::string& error);
};

class Geometry {
 public:
    static Json Bounds(const Rect& value);
    static void Annotate(Json& body, const Layout& layout);
    static void Viewport(Json& body, const Layout& layout);
    static void Overlaps(Json& body, const Layout& layout);
    static Result Save(const Graph& flowgraph, const std::string& name, const Flowgraph::Metadata::NodeLayout& layout);
};

class Lookup {
 public:
    static std::string Similar(const std::string& query, const std::vector<std::string>& candidates,
                               const std::string& noun);
    static std::vector<std::string> Blocks(const Graph& flowgraph);
    static std::string Missing(const Graph& flowgraph, const std::string& name,
                               const std::string& label = "Block");
    static std::vector<std::string> Types();
};

class Validation {
 public:
    static Json Check(const Graph& flowgraph, const Json& args);
    static Json Health(const Json& issues, U64 total);
    static std::string Key(const Json& issue);
};

class Serialize {
 public:
    static Result Collect(const Graph& flowgraph, std::vector<Flowgraph::View::BlockData>& blocks);
    static bool HasPort(const std::vector<Flowgraph::View::InterfaceEntry>& entries, const std::string& port);

    static Json PortNames(const std::vector<Flowgraph::View::InterfaceEntry>& entries);
    static Json Port(const std::string& block, const std::string& port);
    static Json Connection(const std::string& sourceBlock, const std::string& sourcePort,
                           const std::string& destinationBlock, const std::string& destinationPort);
    static Json Inputs(const Flowgraph::View::BlockData& data);
    static std::optional<Json> PendingConfigOptions(const Flowgraph::View::BlockData& data);

    static Json Map(const Parser::Map& map);
    static Json CompactMap(const Parser::Map& map);
    static Json Interface(const std::vector<Flowgraph::View::InterfaceEntry>& entries, bool help = true);
    static Json ConfigOptions(const std::vector<Flowgraph::View::InterfaceEntry>& entries);

    static Json Node(const Graph& flowgraph, const std::string& name);
    static Json NodeLayout(const Graph& flowgraph, const std::string& name);

    static Json BlockDetail(const Graph& flowgraph, const Flowgraph::View::BlockData& data, bool full);
    static Json FlowgraphDetail(const Graph& flowgraph, bool full = false);
    static Json Implementations(const std::string& type);
};

}  // namespace Jetstream::Agent::Detail

#endif  // JETSTREAM_AGENT_TOOLS_SUPPORT_BASE_HH
