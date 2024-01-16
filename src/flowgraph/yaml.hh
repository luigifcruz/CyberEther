#include "jetstream/flowgraph.hh"

#include "rapidyaml.hh"

namespace Jetstream {

struct Flowgraph::YamlImpl {
    ryml::Tree data;

    Parser::Record solveLocalPlaceholder(const Nodes& nodes, const ryml::ConstNodeRef& node);

    // Helper Methods

    static std::vector<std::string> GetMissingKeys(const std::unordered_map<std::string, ryml::ConstNodeRef>& m,
                                                   const std::vector<std::string>& v);
    static std::string GetParameterContents(const std::string& str);
    static std::vector<std::string> GetParameterNodes(const std::string& str);
    static ryml::ConstNodeRef SolvePlaceholder(const ryml::ConstNodeRef& root, const ryml::ConstNodeRef& node);
    static std::unordered_map<std::string, ryml::ConstNodeRef> GatherNodes(const ryml::ConstNodeRef& root,
                                                                           const ryml::ConstNodeRef& node,
                                                                           const std::vector<std::string>& keys,
                                                                           const bool& acceptLess = false);
    static ryml::ConstNodeRef GetNode(const ryml::ConstNodeRef& root, const ryml::ConstNodeRef& node, const std::string& key);
    static bool HasNode(const ryml::ConstNodeRef&, const ryml::ConstNodeRef& node, const std::string& key);
    static std::string ResolveReadable(const ryml::ConstNodeRef& var, const bool& optional = false);
    static std::string ResolveReadableKey(const ryml::ConstNodeRef& var);
};

}  // namespace Jetstream