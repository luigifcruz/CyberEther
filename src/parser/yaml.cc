#include "jetstream/instance.hh"

namespace Jetstream {

std::vector<std::string> Parser::SplitString(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> result;
    size_t pos = 0;
    size_t lastPos = 0;
    while ((pos = str.find(delimiter, lastPos)) != std::string::npos) {
        result.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = pos + delimiter.length();
    }
    result.push_back(str.substr(lastPos));
    return result;
}

std::vector<char> Parser::LoadFile(const std::string& filename) {
    auto filesize = std::filesystem::file_size(filename);
    std::vector<char> data(filesize);
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        JST_ERROR("[PARSER] Can't open configuration file.");
        JST_CHECK_THROW(Result::ERROR);
    }

    file.read(data.data(), filesize);

    if (!file) {
        JST_ERROR("[PARSER] Can't open configuration file.");
        JST_CHECK_THROW(Result::ERROR);
    }

    return data;
}

std::vector<std::string> Parser::GetMissingKeys(const std::unordered_map<std::string, ryml::ConstNodeRef>& m,
                                                const std::vector<std::string>& v) {
    std::vector<std::string> result;
    for (const auto& vkey : v) {
        if (!m.contains(vkey)) {
            result.push_back(vkey);
        }
    }
    return result;
}

std::string Parser::GetParameterContents(const std::string& str) {
    std::regex pattern(R"(\$\{(.*?)\})");
    std::smatch match;
    if (std::regex_search(str, match, pattern) && match.size() > 1) {
        return match.str(1);
    }
    return "";
}

std::vector<std::string> Parser::GetParameterNodes(const std::string& str) {
    return Parser::SplitString(str, ".");
}

ryml::ConstNodeRef Parser::SolvePlaceholder(const ryml::ConstNodeRef& root, const ryml::ConstNodeRef& node) {
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

std::unordered_map<std::string, ryml::ConstNodeRef> Parser::GatherNodes(const ryml::ConstNodeRef& root,
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

ryml::ConstNodeRef Parser::GetNode(const ryml::ConstNodeRef& root, const ryml::ConstNodeRef& node, const std::string& key) {
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

bool Parser::HasNode(const ryml::ConstNodeRef&, const ryml::ConstNodeRef& node, const std::string& key) {
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
std::string Parser::ResolveReadable(const ryml::ConstNodeRef& var, const bool& optional) {
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

std::string Parser::ResolveReadableKey(const ryml::ConstNodeRef& var) {
    if (var.has_key()) {
        return std::string(var.key().str, var.key().len);
    }

    JST_ERROR("[PARSER] Node key not readable.");
    throw Result::ERROR;
}

Parser::Record Parser::SolveLocalPlaceholder(Instance& instance, const ryml::ConstNodeRef& node) {
    if (!node.has_val()) {
        return {ResolveReadable(node)};
    }

    std::string key = std::string(node.val().str, node.val().len);

    std::regex placeholderPattern(R"(\$\{.*\})");
    if (!std::regex_match(key, placeholderPattern)) {
        return {ResolveReadable(node)};
    }

    std::vector<std::string> patternNodes = GetParameterNodes(GetParameterContents(key));

    if (patternNodes.size() != 4) {
        JST_ERROR("[PARSER] Variable {} not found.", key);
        JST_CHECK_THROW(Result::ERROR);
    }

    const auto& graphKey = patternNodes[0];
    const auto& moduleKey = patternNodes[1];
    const auto& arrayKey = patternNodes[2];
    const auto& elementKey = patternNodes[3];

    if (graphKey.compare("graph")) {
        JST_ERROR("[PARSER] Invalid variable {}.", key);
        JST_CHECK_THROW(Result::ERROR);
    }

    if (instance.countBlockState({moduleKey}) == 0) {
        JST_ERROR("[PARSER] Module from the variable {} not found.", key);
        JST_CHECK_THROW(Result::ERROR);
    }
    auto& module = instance.getBlockState(moduleKey).record;

    if (!arrayKey.compare("input")) {
        if (!module.inputMap.contains(elementKey)) {
            JST_ERROR("[PARSER] Input element from the variable {} not found.", key);
            JST_CHECK_THROW(Result::ERROR);
        }
        return {module.inputMap[elementKey].object};
    }

    if (!arrayKey.compare("output")) {
        if (!module.outputMap.contains(elementKey)) {
            JST_ERROR("[PARSER] Output element from the variable {} not found.", key);
            JST_CHECK_THROW(Result::ERROR);
        }
        return {module.outputMap[elementKey].object};
    }

    if (!arrayKey.compare("interface")) {
        if (!module.interfaceMap.contains(elementKey)) {
            JST_ERROR("[PARSER] Interface element from the variable {} not found.", key);
            JST_CHECK_THROW(Result::ERROR);
        }
        return {module.interfaceMap[elementKey].object};
    }

    JST_ERROR("[PARSER] Invalid module array {}. It should be 'input', 'output', or 'interface'.", key);
    throw Result::ERROR;
}

}  // namespace Jetstream
