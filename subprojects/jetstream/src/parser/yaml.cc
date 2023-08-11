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
        JST_FATAL("[PARSER] Can't open configuration file.");
        JST_CHECK_THROW(Result::ERROR);
    }

    file.read(data.data(), filesize);

    if (!file) {
        JST_FATAL("[PARSER] Can't open configuration file.");
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
        JST_FATAL("[PARSER] Node isn't a map.");
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
        JST_FATAL("[PARSER] Failed to parse configuration file due to missing keys: {}",
                  GetMissingKeys(values, keys));
        JST_CHECK_THROW(Result::ERROR);
    }

    return values;
}

ryml::ConstNodeRef Parser::GetNode(const ryml::ConstNodeRef& root, const ryml::ConstNodeRef& node, const std::string& key) {
    if (!node.is_map()) {
        JST_FATAL("[PARSER] Node isn't a map.");
        JST_CHECK_THROW(Result::ERROR);
    }

    for (auto const& child : node.children()) {
        if (child.key() == key) {
            return SolvePlaceholder(root, node[child.key()]);
        }
    }

    JST_FATAL("[PARSER] Node ({}) doesn't exist.", key);
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
std::string Parser::ResolveReadable(const ryml::ConstNodeRef& var) {
    std::string readableVar;
    
    if (var.is_map()) {
        JST_FATAL("[PARSER] Node is a map.");
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

    JST_FATAL("[PARSER] Node value not readable.");
    throw Result::ERROR;
}

std::string Parser::ResolveReadableKey(const ryml::ConstNodeRef& var) {
    if (var.has_key()) {
        return std::string(var.key().str, var.key().len);
    }

    JST_FATAL("[PARSER] Node key not readable.");
    throw Result::ERROR;
}

std::any Parser::SolveLocalPlaceholder(Instance& instance, const ryml::ConstNodeRef& node) {
    if (!node.has_val()) {
        return ResolveReadable(node); 
    }

    std::string key = std::string(node.val().str, node.val().len);

    std::regex placeholderPattern(R"(\$\{.*\})");
    if (!std::regex_match(key, placeholderPattern)) {
        return ResolveReadable(node);
    }

    std::vector<std::string> patternNodes = GetParameterNodes(GetParameterContents(key));

    if (patternNodes.size() != 4) {
        JST_FATAL("[PARSER] Variable {} not found.", key);
        JST_CHECK_THROW(Result::ERROR);
    }

    const auto& graphKey = patternNodes[0];
    const auto& moduleKey = patternNodes[1];
    const auto& arrayKey = patternNodes[2];
    const auto& elementKey = patternNodes[3];

    if (graphKey.compare("graph")) {
        JST_FATAL("[PARSER] Invalid variable {}.", key);
        JST_CHECK_THROW(Result::ERROR);
    }

    if (instance.countBlockState(moduleKey) == 0) {
        JST_FATAL("[PARSER] Module from the variable {} not found.", key);
        JST_CHECK_THROW(Result::ERROR);
    }
    auto& module = instance.getBlockState(moduleKey).record;

    if (!arrayKey.compare("input")) {
        if (!module.data.inputMap.contains(elementKey)) {
            JST_FATAL("[PARSER] Input element from the variable {} not found.", key);
            JST_CHECK_THROW(Result::ERROR);
        }
        return module.data.inputMap[elementKey].object;
    }
    
    if (!arrayKey.compare("output")) {
        if (!module.data.outputMap.contains(elementKey)) {
            JST_FATAL("[PARSER] Output element from the variable {} not found.", key);
            JST_CHECK_THROW(Result::ERROR);
        }
        return module.data.outputMap[elementKey].object;
    }

    if (!arrayKey.compare("interface")) {
        if (!module.data.interfaceMap.contains(elementKey)) {
            JST_FATAL("[PARSER] Interface element from the variable {} not found.", key);
            JST_CHECK_THROW(Result::ERROR);
        }
        return module.data.interfaceMap[elementKey].object;
    }

    JST_FATAL("[PARSER] Invalid module array {}. It should be 'input', 'output', or 'interface'.", key);
    throw Result::ERROR;
}

}  // namespace Jetstream