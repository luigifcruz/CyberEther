#include "support/base.hh"
#include "jetstream/flowgraph_environment.hh"
#include "tools/text.hh"

#include <iterator>

namespace Jetstream::Agent::Detail {
namespace {

void AppendTruncatedValue(Json& entry, const std::string& value, U64 maxLength) {
    if (value.size() > maxLength) {
        entry["value"] = Text::Utf8Prefix(value, maxLength);
        entry["truncated"] = true;
        entry["full_length"] = value.size();
    } else {
        entry["value"] = value;
    }
}

std::string ValueOutline(const std::any& value, U64 depth = 0);

std::string MapOutline(const Parser::Map& map, U64 depth = 0) {
    if (!map.empty() && depth >= 4) {
        return jst::fmt::format("map[{}]", map.size());
    }

    return Value::MapToString(map, [depth](const std::any& value) {
        return ValueOutline(value, depth + 1);
    });
}

std::string ValueOutline(const std::any& value, U64 depth) {
    if (!value.has_value()) {
        return "null";
    }

    if (value.type() == typeid(Parser::Map)) {
        return MapOutline(std::any_cast<const Parser::Map&>(value), depth);
    }

    if (value.type() == typeid(Parser::Sequence)) {
        const auto& sequence = std::any_cast<const Parser::Sequence&>(value);
        if (sequence.empty()) {
            return "seq[0]";
        }
        return jst::fmt::format("seq[{}] of {}", sequence.size(), ValueOutline(sequence.front(), depth + 1));
    }

    if (value.type() == typeid(std::string)) {
        const auto& text = std::any_cast<const std::string&>(value);
        if (text.size() <= 48) {
            return jst::fmt::format("\"{}\"", text);
        }
        return jst::fmt::format("string({} chars)", text.size());
    }

    std::string outline;
    const bool vector = VectorTypes::Visit(value, [&](const auto& entries) {
        using T = typename std::decay_t<decltype(entries)>::value_type;
        outline = jst::fmt::format("{}[{}]", VectorTypes::Name<T>(), entries.size());
    });

    return vector ? outline : Value::AnyToString(value);
}

struct PathStep {
    bool isIndex;
    std::string key;
    U64 index;
};

bool ParseIndexStep(const std::string& path, U64& i, std::vector<PathStep>& steps, std::string& error) {
    U64 j = i + 1;
    U64 index = 0;
    bool digits = false;
    while (j < path.size() && std::isdigit(static_cast<unsigned char>(path[j]))) {
        index = index * 10 + static_cast<U64>(path[j] - '0');
        digits = true;
        ++j;
    }

    if (!digits || j >= path.size() || path[j] != ']') {
        error = jst::fmt::format("Malformed index in path '{}'. Indexes look like 'bursts[3].snr'.", path);
        return false;
    }

    steps.push_back({true, {}, index});
    i = j + 1;
    return true;
}

void ParseKeyStep(const std::string& path, U64& i, std::vector<PathStep>& steps) {
    U64 j = i;
    while (j < path.size() && path[j] != '.' && path[j] != '[') {
        ++j;
    }

    steps.push_back({false, path.substr(i, j - i), 0});
    i = j;
}

bool ParseValuePath(const std::string& path, std::vector<PathStep>& steps, std::string& error) {
    U64 i = 0;
    while (i < path.size()) {
        const char c = path[i];
        if (c == '.') {
            ++i;
        } else if (c == '[') {
            if (!ParseIndexStep(path, i, steps, error)) {
                return false;
            }
        } else {
            ParseKeyStep(path, i, steps);
        }
    }

    if (steps.empty()) {
        error = jst::fmt::format("Path '{}' is empty.", path);
        return false;
    }

    return true;
}

bool DescendKey(const Parser::Map& root,
                const PathStep& step,
                const std::string& location,
                const std::any*& cursor,
                std::string& error) {
    const Parser::Map* map = nullptr;
    if (cursor == nullptr) {
        map = &root;
    } else if (cursor->type() == typeid(Parser::Map)) {
        map = &std::any_cast<const Parser::Map&>(*cursor);
    } else {
        error = jst::fmt::format("{} is not a map, cannot descend into '{}'.", location, step.key);
        return false;
    }

    if (!map->contains(step.key)) {
        std::vector<std::string> available;
        for (const auto& entry : *map) {
            available.push_back(entry.key);
        }

        error = jst::fmt::format("Key '{}' not found at {}. Available keys: {}",
                                 step.key, location, Json(available).dump());
        return false;
    }

    cursor = &map->at(step.key);
    return true;
}

bool DescendIndex(const PathStep& step,
                  const std::string& location,
                  std::any& scratch,
                  const std::any*& cursor,
                  std::string& error) {
    if (cursor == nullptr) {
        error = "The value root is a map; start the path with a key.";
        return false;
    }

    U64 size = 0;
    bool inRange = false;
    if (cursor->type() == typeid(Parser::Sequence)) {
        const auto& sequence = std::any_cast<const Parser::Sequence&>(*cursor);
        size = sequence.size();
        inRange = step.index < size;
        if (inRange) {
            cursor = &sequence[step.index];
        }
    } else {
        const bool vector = VectorTypes::Visit(*cursor, [&](const auto& entries) {
            using T = typename std::decay_t<decltype(entries)>::value_type;
            size = entries.size();
            inRange = step.index < size;
            if (inRange) {
                scratch = T(entries[step.index]);
            }
        });

        if (!vector) {
            error = jst::fmt::format("{} is not a sequence, cannot index it.", location);
            return false;
        }

        if (inRange) {
            cursor = &scratch;
        }
    }

    if (!inRange) {
        error = jst::fmt::format("{} has {} element{}; index {} is out of range.",
                                 location, size, size == 1 ? "" : "s", step.index);
        return false;
    }

    return true;
}

bool ResolveValuePath(const Parser::Map& root, const std::string& path, std::any& resolved, std::string& error) {
    std::vector<PathStep> steps;
    if (!ParseValuePath(path, steps, error)) {
        return false;
    }

    std::any scratch;
    const std::any* cursor = nullptr;
    std::string walked;
    for (const auto& step : steps) {
        const std::string location = walked.empty() ? "the value root" : jst::fmt::format("'{}'", walked);

        if (step.isIndex) {
            if (!DescendIndex(step, location, scratch, cursor, error)) {
                return false;
            }
            jst::fmt::format_to(std::back_inserter(walked), "[{}]", step.index);
        } else {
            if (!DescendKey(root, step, location, cursor, error)) {
                return false;
            }
            jst::fmt::format_to(std::back_inserter(walked), "{}{}", walked.empty() ? "" : ".", step.key);
        }
    }

    resolved = *cursor;
    return true;
}

std::string EncodeMap(const Parser::Map& map) {
    std::string encoded;
    if (Parser::YamlEncode(map, encoded) != Result::SUCCESS) {
        encoded = Value::MapToString(map);
    }
    return encoded;
}

Json ListEnvironment(const Graph& flowgraph, const Json& args) {
    const std::string filter = Text::ToLower(args.value("query", std::string{}));

    std::vector<std::string> keys;
    if (flowgraph->environment().keys(keys) != Result::SUCCESS) {
        return Response::Error("Could not list the flowgraph environment.");
    }
    std::sort(keys.begin(), keys.end());

    Json entries = Json::array();
    for (const auto& key : keys) {
        if (!Value::KeyMatches(key, filter)) {
            continue;
        }

        Parser::Map value;
        if (flowgraph->environment().get(key, value) != Result::SUCCESS) {
            continue;
        }

        Json entry = {{"key", key}};
        AppendTruncatedValue(entry, MapOutline(value), 512);
        entries.push_back(std::move(entry));
    }

    return Json({
        {"shown", entries.size()},
        {"total", keys.size()},
        {"entries", std::move(entries)},
    });
}

Json InspectEnvironment(const Graph& flowgraph, const Json& args) {
    if (!flowgraph) {
        return Response::Error("No flowgraph is currently selected.");
    }

    if (!args.contains("key")) {
        return ListEnvironment(flowgraph, args);
    }

    const std::string key = args.value("key", std::string{});
    if (key.empty()) {
        return Response::Error("'key' is required.");
    }

    std::vector<std::string> keys;
    const bool listed = flowgraph->environment().keys(keys) == Result::SUCCESS;
    if (!listed || std::find(keys.begin(), keys.end(), key) == keys.end()) {
        return Response::Error(jst::fmt::format("Environment key '{}' not found.{}",
                                                key, Lookup::Similar(key, keys, "environment keys")));
    }

    Parser::Map value;
    if (flowgraph->environment().get(key, value) != Result::SUCCESS) {
        return Response::Error(jst::fmt::format("Could not read environment key '{}'.", key));
    }

    const std::string path = args.value("path", std::string{});
    std::string encoded;
    if (path.empty()) {
        encoded = EncodeMap(value);
    } else {
        std::any resolved;
        std::string pathError;
        if (!ResolveValuePath(value, path, resolved, pathError)) {
            return Response::Error(pathError);
        }

        if (resolved.type() == typeid(Parser::Map)) {
            encoded = EncodeMap(std::any_cast<const Parser::Map&>(resolved));
        } else {
            encoded = Value::AnyToString(resolved);
        }
    }

    Json j = {{"key", key}};
    if (!path.empty()) {
        j["path"] = path;
    }
    AppendTruncatedValue(j, encoded, Arguments::Clamp(args, "max_length", 8192, 256, 262144));

    return j;
}

}  // namespace

std::vector<Entry> Toolbox::Environment() {
    return {Entry::FromJson({
        "inspect_environment",
        "Discover or read runtime environment data. Omit key to list entries whose values are structural outlines, not contents. Provide key and optional path to read the contents of a specific value or subtree. A truncated value reports its full_length; a narrower path or a larger max_length returns more of it.",
        R"({"type":"object","properties":{"query":{"type":"string","description":"Optional key filter when listing."},"key":{"type":"string"},"path":{"type":"string","description":"Dot path with [n] sequence indexes."},"max_length":{"type":"integer","minimum":256,"maximum":262144}},"additionalProperties":false})",
        {.needsGraph = true},
    }, InspectEnvironment)};
}

}  // namespace Jetstream::Agent::Detail
