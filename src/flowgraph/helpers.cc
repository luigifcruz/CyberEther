#include <ranges>
#include <regex>

#include "jetstream/flowgraph.hh"

namespace Jetstream {

std::vector<std::string> Flowgraph::GetMissingKeys(const std::unordered_map<std::string, ryml::ConstNodeRef>& m,
                                                const std::vector<std::string>& v) {
    std::vector<std::string> result;
    for (const auto& vkey : v) {
        if (!m.contains(vkey)) {
            result.push_back(vkey);
        }
    }
    return result;
}

std::string Flowgraph::GetParameterContents(const std::string& str) {
    std::regex pattern(R"(\$\{(.*?)\})");
    std::smatch match;
    if (std::regex_search(str, match, pattern) && match.size() > 1) {
        return match.str(1);
    }
    return "";
}

std::vector<std::string> Flowgraph::GetParameterNodes(const std::string& str) {
    return Parser::SplitString(str, ".");
}

ryml::ConstNodeRef Flowgraph::SolvePlaceholder(const ryml::ConstNodeRef& root, const ryml::ConstNodeRef& node) {
    if (!node.has_val()) {
        return node;
    }

    std::string key = std::string(node.val().str, node.val().len);

    std::regex placeholderPattern(R"(\$\{.*\})");
    if (!std::regex_match(key, placeholderPattern)) {
        return node;
    }

    if (!root.is_map()) {
        return node;
    }

    U64 depth = 0;
    ryml::ConstNodeRef currentNode = root;
    std::vector<std::string> patternNodes = GetParameterNodes(GetParameterContents(key));

    for (const auto& nextNode : patternNodes) {
        for (const auto& child : currentNode.children()) {
            if (child.key() == nextNode) {
                currentNode = currentNode[child.key()];
                depth += 1;
            }
        }
    }

    if (depth != patternNodes.size()) {
        currentNode = node;
    }

    return currentNode;
}

std::unordered_map<std::string, ryml::ConstNodeRef> Flowgraph::GatherNodes(const ryml::ConstNodeRef& root,
                                                                        const ryml::ConstNodeRef& node,
                                                                        const std::vector<std::string>& keys,
                                                                        const bool& acceptLess) {
    std::unordered_map<std::string, ryml::ConstNodeRef> values;

    if (!node.is_map()) {
        JST_ERROR("[PARSER] Node isn't a map.");
        JST_CHECK_THROW(Result::ERROR);
    }

    for (const auto& child : node.children()) {
        for (const auto& key : keys) {
            if (child.key() == key) {
                values[key] = SolvePlaceholder(root, node[child.key()]);
                break;
            }
        }
    }

    if (values.size() != keys.size() && !acceptLess) {
        JST_ERROR("[PARSER] Failed to parse configuration file due to missing keys: {}",
                  GetMissingKeys(values, keys));
        JST_CHECK_THROW(Result::ERROR);
    }

    return values;
}

ryml::ConstNodeRef Flowgraph::GetNode(const ryml::ConstNodeRef& root, const ryml::ConstNodeRef& node, const std::string& key) {
    if (!node.is_map()) {
        JST_ERROR("[PARSER] Node isn't a map.");
        JST_CHECK_THROW(Result::ERROR);
    }

    for (auto const& child : node.children()) {
        if (child.key() == key) {
            return SolvePlaceholder(root, node[child.key()]);
        }
    }

    JST_ERROR("[PARSER] Node ({}) doesn't exist.", key);
    throw Result::ERROR;
}

bool Flowgraph::HasNode(const ryml::ConstNodeRef&, const ryml::ConstNodeRef& node, const std::string& key) {
    if (!node.is_map()) {
        return false;
    }

    for (auto const& child : node.children()) {
        if (child.key() == key) {
            return true;
        }
    }

    return false;
}

// TODO: Sanitize string case.
std::string Flowgraph::ResolveReadable(const ryml::ConstNodeRef& var, const bool& optional) {
    std::string readableVar;

    if (var.is_map()) {
        JST_ERROR("[PARSER] Node is a map.");
        JST_CHECK_THROW(Result::ERROR);
    }

    if (var.num_children()) {
        for (U64 i = 0; i < var.num_children(); i++) {
            std::string partialStr;
            if (var[i].has_val()) {
                var[i] >> partialStr;
            }
            if (i > 0) {
                readableVar += ", ";
            }
            readableVar += partialStr;
        }
        return readableVar;
    }

    if (var.has_val()) {
        var >> readableVar;
        return readableVar;
    }

    if (optional) {
        return "N/A";
    }

    JST_ERROR("[PARSER] Node value not readable.");
    throw Result::ERROR;
}

std::string Flowgraph::ResolveReadableKey(const ryml::ConstNodeRef& var) {
    if (var.has_key()) {
        return std::string(var.key().str, var.key().len);
    }

    JST_ERROR("[PARSER] Node key not readable.");
    throw Result::ERROR;
}

}  // namespace Jetstream
