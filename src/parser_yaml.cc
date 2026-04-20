#include "jetstream/parser.hh"

#include <ryml.hpp>

#include <ryml_std.hpp>

#include <algorithm>

namespace Jetstream {

namespace {

std::string NormalizeScalar(const std::string& value) {
    if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
        return value.substr(1, value.size() - 2);
    }

    return value;
}

Result SerializeMap(const Parser::Map& data, ryml::NodeRef& node);
Result SerializeSequence(const Parser::Sequence& sequence, ryml::NodeRef& node);
Result DeserializeMap(ryml::ConstNodeRef node, Parser::Map& data);
Result DeserializeSequence(ryml::ConstNodeRef node, Parser::Sequence& sequence);

Result SerializeValue(const std::any& value, ryml::NodeRef& node) {
    if (value.type() == typeid(Parser::Map)) {
        return SerializeMap(std::any_cast<const Parser::Map&>(value), node);
    }

    if (value.type() == typeid(Parser::Sequence)) {
        return SerializeSequence(std::any_cast<const Parser::Sequence&>(value), node);
    }

    std::string encoded;
    JST_CHECK(Parser::TypedToString(value, encoded));
    node << encoded;

    if (std::count(encoded.begin(), encoded.end(), '\n')) {
        node |= ryml::VAL_LITERAL;
    }

    return Result::SUCCESS;
}

Result SerializeSequence(const Parser::Sequence& sequence, ryml::NodeRef& node) {
    node |= ryml::SEQ;

    for (const auto& value : sequence) {
        auto child = node.append_child();
        JST_CHECK(SerializeValue(value, child));
    }

    return Result::SUCCESS;
}

Result DeserializeValue(ryml::ConstNodeRef node, std::any& value) {
    if (node.is_map()) {
        Parser::Map nested;
        JST_CHECK(DeserializeMap(node, nested));
        value = std::move(nested);
        return Result::SUCCESS;
    }

    if (node.is_seq()) {
        Parser::Sequence nested;
        JST_CHECK(DeserializeSequence(node, nested));
        value = std::move(nested);
        return Result::SUCCESS;
    }

    if (!node.has_val()) {
        JST_ERROR("[PARSER] YAML node must be a scalar, map, or sequence.");
        return Result::ERROR;
    }

    std::string scalar;
    node >> scalar;
    value = NormalizeScalar(scalar);
    return Result::SUCCESS;
}

Result DeserializeSequence(ryml::ConstNodeRef node, Parser::Sequence& sequence) {
    if (!node.is_seq()) {
        JST_ERROR("[PARSER] YAML node must be a sequence.");
        return Result::ERROR;
    }

    Parser::Sequence decoded;
    decoded.reserve(node.num_children());

    for (const auto& child : node.children()) {
        std::any value;
        JST_CHECK(DeserializeValue(child, value));
        decoded.push_back(std::move(value));
    }

    sequence = std::move(decoded);
    return Result::SUCCESS;
}

Result SerializeMap(const Parser::Map& data, ryml::NodeRef& node) {
    node |= ryml::MAP;

    for (const auto& [key, value] : data) {
        if (key.empty()) {
            continue;
        }

        std::string encoded;
        const bool isScalar = value.type() != typeid(Parser::Map) && value.type() != typeid(Parser::Sequence);
        if (isScalar) {
            JST_CHECK(Parser::TypedToString(value, encoded));
        }

        auto child = node.append_child();
        child << ryml::key(key);
        JST_CHECK(SerializeValue(value, child));
    }

    return Result::SUCCESS;
}

Result DeserializeMap(ryml::ConstNodeRef node, Parser::Map& data) {
    if (!node.is_map()) {
        JST_ERROR("[PARSER] YAML node must be a map.");
        return Result::ERROR;
    }

    Parser::Map decoded;

    for (const auto& child : node.children()) {
        std::string key(child.key().str, child.key().len);
        std::any value;
        JST_CHECK(DeserializeValue(child, value));
        decoded[key] = std::move(value);
    }

    data = std::move(decoded);
    return Result::SUCCESS;
}

}  // namespace

Result Parser::YamlEncode(const Parser::Map& data, std::string& yaml) {
    ryml::Tree tree;
    auto root = tree.rootref();
    JST_CHECK(SerializeMap(data, root));

    const auto emitted = ryml::emitrs_yaml<std::vector<char>>(tree);
    yaml.assign(emitted.begin(), emitted.end());
    return Result::SUCCESS;
}

Result Parser::YamlDecode(const std::string& yaml, Parser::Map& data) {
    if (yaml.empty()) {
        data.clear();
        return Result::SUCCESS;
    }

    ryml::Tree tree;
    try {
        tree = ryml::parse_in_arena(ryml::to_csubstr(yaml));
    } catch (...) {
        JST_ERROR("[PARSER] Failed to parse YAML document.");
        return Result::ERROR;
    }

    auto root = tree.rootref();
    if (!root.is_map() && root.num_children() > 0) {
        root = root[0];
    }

    return DeserializeMap(root, data);
}

}  // namespace Jetstream
