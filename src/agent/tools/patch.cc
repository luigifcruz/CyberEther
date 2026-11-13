#include "support/base.hh"
#include "tools/text.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/runtime.hh"
#include "tools/naming.hh"

#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <set>
#include <string_view>
#include <unordered_map>

namespace Jetstream::Agent::Detail {
namespace {

constexpr F32 kAppliedTolerance = 2.0f;
constexpr std::string_view kRefAlias = "ref:";
constexpr std::array<const char*, 3> kRefFields = {"name", "source", "block"};

constexpr const char* kOperationsText =
    "'operations' arrived as text that is not a valid JSON array, so nothing was changed. "
    "If a closing bracket or quote was left out, send the complete array again. If an "
    "argument contains a literal </parameter>, </function> or <function= sequence, the call "
    "was cut off there. No tool, patch_block_text included, can receive such a sequence, and "
    "no escaping helps. In code, build it from pieces such as '</' + 'parameter>'. Where the "
    "exact text is required, as in a comment or note, leave that line out, make the rest of "
    "the edit, and tell the user to type that line by hand.";

bool IsRef(const std::string& value) {
    return value.size() > 1 && value.front() == '$';
}

// The exact label wins, so a block created with ref "ref:a" stays reachable as "$ref:a";
// otherwise "$ref:label" is read as "$label".
template<typename Refs>
std::string RefLabel(const Refs& refs, const std::string& value) {
    std::string label = value.substr(1);
    const bool aliased = label.size() > kRefAlias.size() && label.starts_with(kRefAlias);
    if (aliased && !refs.contains(label)) {
        label.erase(0, kRefAlias.size());
    }
    return label;
}

struct Patch {
    const Graph& flowgraph;
    std::unordered_map<std::string, std::string> refs;

    std::string resolve(const Json& operation, const char* key) const {
        std::string value = operation.value(key, std::string{});
        if (!IsRef(value)) {
            return value;
        }

        const auto it = refs.find(RefLabel(refs, value));
        if (it == refs.end()) {
            return {};
        }
        return it->second;
    }

    std::string unresolved(const Json& operation, const char* key, const std::string& label) const {
        const std::string value = operation.value(key, std::string{});
        if (IsRef(value) && !refs.contains(RefLabel(refs, value))) {
            return jst::fmt::format("Ref '{}' is not defined by an earlier create_block in this patch.", value);
        }
        return Lookup::Missing(flowgraph, value, label);
    }
};

enum class FieldKind : U8 {
    String,
    Object,
    Coordinate,
    Size,
};

struct Field {
    const char* name;
    FieldKind kind;
    bool required = false;
    const char* description = nullptr;

    constexpr Field(const char* name, FieldKind kind, bool required = false,
                    const char* description = nullptr) :
        name(name), kind(kind), required(required), description(description) {}

    constexpr Field(const char* name, FieldKind kind, const char* description) :
        name(name), kind(kind), description(description) {}
};

struct Operation {
    const char* name;
    std::initializer_list<Field> fields;
    void (*apply)(Patch& patch, const Json& operation, Json& result);
    std::string (*check)(const Json& operation) = nullptr;
    bool hidden = false;
};

struct Implementation {
    DeviceType device = DeviceType::CPU;
    RuntimeType runtime = RuntimeType::NATIVE;
    ProviderType provider;
};

std::string ResultName(Result result) {
    switch (result) {
        case Result::SUCCESS:
            return "success";
        case Result::RECREATE:
            return "recreated";
        case Result::INCOMPLETE:
            return "incomplete";
        default:
            return "error";
    }
}

const char* FieldType(FieldKind kind) {
    switch (kind) {
        case FieldKind::String:
            return "string";
        case FieldKind::Object:
            return "object";
        default:
            return "number";
    }
}

bool Applied(Result result) {
    return result == Result::SUCCESS || result == Result::RECREATE;
}

std::string ValidateField(const Field& field, const Json& value) {
    switch (field.kind) {
        case FieldKind::String:
            if (!value.is_string()) {
                return jst::fmt::format("field '{}' must be a string.", field.name);
            }
            return {};
        case FieldKind::Object:
            if (!value.is_object()) {
                return jst::fmt::format("field '{}' must be an object.", field.name);
            }
            return {};
        case FieldKind::Coordinate:
        case FieldKind::Size: {
            if (!value.is_number()) {
                return jst::fmt::format("field '{}' must be a number.", field.name);
            }

            const auto number = value.get<F64>();
            if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<F32>::max()) {
                return jst::fmt::format("field '{}' must be a finite graph coordinate.", field.name);
            }
            if (field.kind == FieldKind::Size && number < 0) {
                return jst::fmt::format("field '{}' cannot be negative.", field.name);
            }
            return {};
        }
    }
    return {};
}

std::string CheckLayout(const Json& operation) {
    const bool hasPosition = operation.contains("x") || operation.contains("y");
    const bool hasSize = operation.contains("width") || operation.contains("height");
    if (!hasPosition && !hasSize) {
        return "set_node_layout requires at least one of 'x', 'y', 'width', or 'height'.";
    }
    return {};
}

bool SelectImplementation(const std::string& type, const Json& operation, Implementation& selected, Json& result) {
    const auto targets = Registry::ListAvailableBlockTargets(type);
    const std::string device = Text::ToLower(operation.value("device", std::string{}));
    const std::string runtime = Text::ToLower(operation.value("runtime", std::string{}));
    const std::string provider = Text::ToLower(operation.value("provider", std::string{}));

    const auto matches = [&](const Registry::BlockTarget& target) {
        const bool deviceMatches = device.empty() || Text::ToLower(GetDeviceName(target.device)) == device;
        const bool runtimeMatches = runtime.empty() || Text::ToLower(GetRuntimeName(target.runtime)) == runtime;
        const bool providerMatches = provider.empty() || Text::ToLower(target.provider) == provider;
        return deviceMatches && runtimeMatches && providerMatches;
    };

    const auto found = std::find_if(targets.begin(), targets.end(), matches);
    if (found != targets.end()) {
        selected = {found->device, found->runtime, found->provider};
        return true;
    }

    result["error"] = "Requested implementation is unavailable.";
    result["available_implementations"] = Serialize::Implementations(type);
    return false;
}

void PlaceNode(const Graph& flowgraph, const Json& operation, const std::string& name, Json& result) {
    if (!operation.contains("x") && !operation.contains("y")) {
        return;
    }

    Flowgraph::Metadata::NodeLayout meta;
    flowgraph->metadata().get("node", meta, name);
    meta.x = operation.value("x", meta.x);
    meta.y = operation.value("y", meta.y);

    if (Geometry::Save(flowgraph, name, meta) != Result::SUCCESS) {
        result["layout_error"] = "Could not save the requested position.";
    }
}

void DescribeCreated(const Graph& flowgraph, const std::string& name, const std::string& ref,
                     Result created, Json& result) {
    Flowgraph::View::BlockData data;
    flowgraph->view().block(name, data);

    result.update({
        {"status", ResultName(created)},
        {"name", name},
        {"ref", ref},
        {"config", Serialize::CompactMap(data.config)},
        {"state", GetBlockStateName(data.state)},
        {"input_ports", Serialize::Interface(data.interfaceInputs)},
        {"output_ports", Serialize::Interface(data.interfaceOutputs)},
    });

    if (!data.diagnostic.empty()) {
        result["diagnostic"] = data.diagnostic;
    }

    auto options = Serialize::PendingConfigOptions(data);
    if (options && !options->empty()) {
        result["config_options"] = std::move(*options);
    }
}

void CreateBlock(Patch& patch, const Json& operation, Json& result) {
    const auto& flowgraph = patch.flowgraph;
    const std::string type = operation.value("type", std::string{});
    const std::string ref = operation.value("ref", std::string{});

    if (type.empty() || Registry::ListAvailableBlocks(type).empty()) {
        const auto similar = Lookup::Similar(type, Lookup::Types(), "block types");
        result["error"] = jst::fmt::format("Unknown block type '{}'.{}", type, similar);
        return;
    }

    Implementation selected;
    if (!SelectImplementation(type, operation, selected, result)) {
        return;
    }

    std::string base = operation.value("name", type);
    if (base.empty()) {
        base = type;
    }
    const std::string name = Naming::UniqueName(base, [&](const std::string& candidate) {
        return flowgraph->view().has(candidate);
    });

    Parser::Map config;
    if (operation.contains("config") && Value::FromJson(operation["config"], config) != Result::SUCCESS) {
        result["error"] = jst::fmt::format("Could not decode 'config' for block type '{}'.", type);
        return;
    }

    const Result created = flowgraph->blockCreate(name, type, config, {},
                                                  selected.device, selected.runtime, selected.provider);
    if (created != Result::SUCCESS && created != Result::INCOMPLETE) {
        result["error"] = jst::fmt::format("Could not create block type '{}'.", type);
        return;
    }

    PlaceNode(flowgraph, operation, name, result);
    if (!ref.empty()) {
        patch.refs[ref] = name;
    }
    DescribeCreated(flowgraph, name, ref, created, result);
}

void ReconfigureBlock(Patch& patch, const Json& operation, Json& result) {
    const auto& flowgraph = patch.flowgraph;
    const std::string name = patch.resolve(operation, "name");

    Flowgraph::View::BlockData current;
    if (name.empty() || flowgraph->view().block(name, current) != Result::SUCCESS) {
        result["error"] = patch.unresolved(operation, "name", "Block");
        return;
    }

    Parser::Map config;
    if (Value::FromJson(operation["config"], config) != Result::SUCCESS) {
        result["error"] = jst::fmt::format("Could not decode 'config' for block '{}'.", name);
        return;
    }

    const Result changed = flowgraph->blockReconfigure(name, config);
    if (!Applied(changed)) {
        result["error"] = jst::fmt::format("Could not reconfigure block '{}'.", name);
        return;
    }

    Flowgraph::View::BlockData after;
    flowgraph->view().block(name, after);

    Parser::Map applied;
    for (const auto& field : config) {
        if (after.config.contains(field.key)) {
            applied[field.key] = after.config.at(field.key);
        }
    }

    result.update({
        {"status", ResultName(changed)},
        {"name", name},
        {"config", Serialize::CompactMap(applied)},
        {"state", GetBlockStateName(after.state)},
    });

    if (!after.diagnostic.empty()) {
        result["diagnostic"] = after.diagnostic;
    }
}

void EditText(const Graph& flowgraph, const std::string& name, const std::string& field,
              const std::string& patchText, Json& result) {
    std::string text;
    if (const auto error = BlockText::Read(flowgraph, name, field, text); !error.empty()) {
        result["error"] = error;
        return;
    }

    auto patched = BlockText::Apply(text, patchText);
    if (!patched.error.empty()) {
        result["error"] = jst::fmt::format("{} Nothing was changed.", patched.error);
        return;
    }

    Parser::Map config;
    config[field] = patched.text;

    const Result changed = flowgraph->blockReconfigure(name, config);
    if (!Applied(changed)) {
        result["error"] = jst::fmt::format("Could not apply the patched '{}' to block '{}'.", field, name);
        return;
    }

    Flowgraph::View::BlockData after;
    flowgraph->view().block(name, after);

    result.update({
        {"status", ResultName(changed)},
        {"name", name},
        {"field", field},
        {"line_count", patched.lineCount},
        {"changes", std::move(patched.changes)},
        {"state", GetBlockStateName(after.state)},
    });

    if (!after.diagnostic.empty()) {
        result["diagnostic"] = after.diagnostic;
    }
    if (!patched.note.empty()) {
        result["note"] = patched.note;
    }
}

void PatchBlockText(Patch& patch, const Json& operation, Json& result) {
    EditText(patch.flowgraph, patch.resolve(operation, "name"), operation.value("field", std::string{}),
             operation.value("patch", std::string{}), result);
}

void Connect(Patch& patch, const Json& operation, Json& result) {
    const auto& flowgraph = patch.flowgraph;
    const std::string source = patch.resolve(operation, "source");
    const std::string block = patch.resolve(operation, "block");
    const std::string output = operation.value("output", std::string{});
    const std::string input = operation.value("input", std::string{});

    Flowgraph::View::BlockData sourceData, destinationData;
    if (flowgraph->view().block(source, sourceData) != Result::SUCCESS) {
        result["error"] = patch.unresolved(operation, "source", "Source block");
        return;
    }
    if (flowgraph->view().block(block, destinationData) != Result::SUCCESS) {
        result["error"] = patch.unresolved(operation, "block", "Destination block");
        return;
    }
    if (!Serialize::HasPort(sourceData.interfaceOutputs, output)) {
        result["error"] = jst::fmt::format("Unknown output port '{}'.", output);
        result["available_outputs"] = Serialize::PortNames(sourceData.interfaceOutputs);
        return;
    }
    if (!Serialize::HasPort(destinationData.interfaceInputs, input)) {
        result["error"] = jst::fmt::format("Unknown input port '{}'.", input);
        result["available_inputs"] = Serialize::PortNames(destinationData.interfaceInputs);
        return;
    }

    const Result connected = flowgraph->blockConnect(block, input, source, output);
    if (!Applied(connected)) {
        result["error"] = jst::fmt::format("Could not connect {}.{} to {}.{}.", source, output, block, input);
        return;
    }

    result.update({
        {"status", ResultName(connected)},
        {"connection", Serialize::Connection(source, output, block, input)},
    });
}

void Disconnect(Patch& patch, const Json& operation, Json& result) {
    const std::string block = patch.resolve(operation, "block");
    const std::string input = operation.value("input", std::string{});

    if (block.empty() || !patch.flowgraph->view().has(block)) {
        result["error"] = patch.unresolved(operation, "block", "Block");
        return;
    }

    const Result disconnected = patch.flowgraph->blockDisconnect(block, input);
    if (!Applied(disconnected)) {
        result["error"] = jst::fmt::format("Could not disconnect '{}.{}'.", block, input);
        return;
    }

    result.update({
        {"status", ResultName(disconnected)},
        {"block", block},
        {"input", input},
    });
}

void DeleteBlock(Patch& patch, const Json& operation, Json& result) {
    const std::string name = patch.resolve(operation, "name");

    if (name.empty() || !patch.flowgraph->view().has(name)) {
        result["error"] = patch.unresolved(operation, "name", "Block");
        return;
    }

    if (patch.flowgraph->blockDestroy(name) != Result::SUCCESS) {
        result["error"] = jst::fmt::format("Could not delete block '{}'.", name);
        return;
    }

    result.update({
        {"status", "success"},
        {"name", name},
    });
}

void SetNodeLayout(Patch& patch, const Json& operation, Json& result) {
    const auto& flowgraph = patch.flowgraph;
    const std::string name = patch.resolve(operation, "name");

    Flowgraph::View::BlockData current;
    if (name.empty() || flowgraph->view().block(name, current) != Result::SUCCESS) {
        result["error"] = patch.unresolved(operation, "name", "Block");
        return;
    }

    Flowgraph::Metadata::NodeLayout meta;
    flowgraph->metadata().get("node", meta, name);

    const auto applyField = [&](const char* key, F32& target) {
        if (operation.contains(key)) {
            target = operation[key].get<F32>();
        }
    };
    applyField("x", meta.x);
    applyField("y", meta.y);
    applyField("width", meta.width);
    applyField("height", meta.height);

    if (Geometry::Save(flowgraph, name, meta) != Result::SUCCESS) {
        result["error"] = jst::fmt::format("Could not set node layout for block '{}'.", name);
        return;
    }

    result.update({
        {"status", "success"},
        {"name", name},
        {"node", Serialize::Node(flowgraph, name)},
    });
}

const std::array<Operation, 7> kOperations = {{
    {"create_block", {
        {"type", FieldKind::String, true, "Exact block type from search_blocks or get_block_docs."},
        {"ref", FieldKind::String, "Label of your choice for this new block. Later operations in this call name "
                                   "the block as '$' plus the label: ref 'src' is written '$src'."},
        {"name", FieldKind::String, "Block name; generated from the type when omitted."},
        {"device", FieldKind::String, "Compute device implementation, such as cpu or cuda. Default cpu."},
        {"runtime", FieldKind::String, "Runtime implementation, such as native or python. Default native."},
        {"provider", FieldKind::String, "Provider implementation from search_blocks. Default generic."},
        {"config", FieldKind::Object, "Config fields to set, in the base values shown by get_block_docs default_config."},
        {"x", FieldKind::Coordinate, "Initial left edge in graph coordinates."},
        {"y", FieldKind::Coordinate, "Initial top edge in graph coordinates."},
    }, CreateBlock},
    {"reconfigure_block", {
        {"name", FieldKind::String, true, "Exact block name from inspect_flowgraph."},
        {"config", FieldKind::Object, true, "Config fields to change, in base values; omitted fields stay unchanged."},
    }, ReconfigureBlock},
    {"patch_block_text", {
        {"name", FieldKind::String, true, "Exact block name from inspect_flowgraph."},
        {"field", FieldKind::String, true, "Text config field, such as code."},
        {"patch", FieldKind::String, true, "Hunks of @@, space, -, and + lines."},
    }, PatchBlockText, nullptr, true},
    {"connect", {
        {"source", FieldKind::String, true, "Exact name of the upstream block, or '$' plus a ref from this call."},
        {"output", FieldKind::String, true, "Output port of the upstream block."},
        {"block", FieldKind::String, true, "Exact name of the downstream block, or '$' plus a ref from this call."},
        {"input", FieldKind::String, true, "Input port of the downstream block."},
    }, Connect},
    {"disconnect", {
        {"block", FieldKind::String, true, "Exact name of the downstream block."},
        {"input", FieldKind::String, true, "Input port to disconnect from its source."},
    }, Disconnect},
    {"delete_block", {
        {"name", FieldKind::String, true, "Exact block name from inspect_flowgraph."},
    }, DeleteBlock},
    {"set_node_layout", {
        {"name", FieldKind::String, true, "Exact block name from inspect_flowgraph."},
        {"x", FieldKind::Coordinate, "New left edge in graph coordinates."},
        {"y", FieldKind::Coordinate, "New top edge in graph coordinates."},
        {"width", FieldKind::Size, "Saved content width in graph units."},
        {"height", FieldKind::Size, "Saved content height in graph units."},
    }, SetNodeLayout, CheckLayout},
}};

std::string ValidateOperation(const Operation& entry, const Json& operation) {
    for (const auto& [key, value] : operation.items()) {
        if (key == "op") {
            continue;
        }

        const auto field = std::find_if(entry.fields.begin(), entry.fields.end(), [&](const Field& candidate) {
            return key == candidate.name;
        });
        if (field == entry.fields.end()) {
            return jst::fmt::format("has unknown field '{}'.", key);
        }
        if (auto error = ValidateField(*field, value); !error.empty()) {
            return jst::fmt::format("{} {}", entry.name, error);
        }
    }

    for (const auto& field : entry.fields) {
        if (field.required && !operation.contains(field.name)) {
            return jst::fmt::format("{} requires field '{}'.", entry.name, field.name);
        }
    }

    if (!entry.check) {
        return {};
    }
    return entry.check(operation);
}

std::string PatchSchema() {
    Json variants = Json::array();
    for (const auto& entry : kOperations) {
        if (entry.hidden) {
            continue;
        }

        Json properties = {{"op", {{"const", entry.name}}}};
        Json required = Json::array({"op"});

        for (const auto& field : entry.fields) {
            Json property = {{"type", FieldType(field.kind)}};
            if (field.description) {
                property["description"] = field.description;
            }
            properties[field.name] = std::move(property);

            if (field.required) {
                required.push_back(field.name);
            }
        }

        variants.push_back({
            {"type", "object"},
            {"required", std::move(required)},
            {"properties", std::move(properties)},
            {"additionalProperties", false},
        });
    }

    const Json flowgraphName = {
        {"type", "string"},
        {"description", "Name used only when no graph is selected and one must be created."},
    };
    const Json operations = {
        {"type", "array"},
        {"description", "Ordered operations as a JSON array of objects, never a string. Each object "
                        "has 'op' and the fields of that operation."},
        {"minItems", 1},
        {"items", {{"oneOf", std::move(variants)}}},
    };

    return Json({
        {"type", "object"},
        {"required", Json::array({"operations"})},
        {"properties", {
            {"flowgraph_name", flowgraphName},
            {"operations", operations},
        }},
        {"additionalProperties", false},
    }).dump();
}

const Operation* FindOperation(const std::string& name) {
    const auto found = std::find_if(kOperations.begin(), kOperations.end(), [&](const Operation& operation) {
        return name == operation.name;
    });
    if (found == kOperations.end()) {
        return nullptr;
    }
    return &*found;
}

std::set<std::string> TouchedBlocks(const Json& results) {
    std::set<std::string> touched;
    for (const auto& result : results) {
        for (const char* key : {"name", "block"}) {
            if (result.contains(key) && result[key].is_string()) {
                touched.insert(result[key].get<std::string>());
            }
        }

        if (result.contains("connection")) {
            touched.insert(result["connection"]["destination"].value("block", std::string{}));
        }
    }
    return touched;
}

Json PatchHealth(const Graph& flowgraph, const Json& results, const std::set<std::string>& before) {
    const auto health = Validation::Check(flowgraph, Json::object());
    if (!health.contains("issues")) {
        return health;
    }

    const auto touched = TouchedBlocks(results);
    Json shown = Json::array();
    for (const auto& issue : health["issues"]) {
        const bool onTouched = touched.contains(issue.value("block", std::string{}));
        const bool isNew = !before.contains(Validation::Key(issue));
        if (onTouched || isNew) {
            shown.push_back(issue);
        }
    }

    return Validation::Health(shown, health["issues"].size());
}

std::set<std::string> IssueKeys(const Graph& flowgraph) {
    std::set<std::string> keys;

    const auto health = Validation::Check(flowgraph, Json::object());
    if (!health.contains("issues")) {
        return keys;
    }

    for (const auto& issue : health["issues"]) {
        keys.insert(Validation::Key(issue));
    }
    return keys;
}

Json PatchResponse(const Graph& flowgraph, Json results, const char* status, const std::set<std::string>& before) {
    for (auto& result : results) {
        if (!result.contains("name")) {
            continue;
        }

        const auto name = result["name"].get<std::string>();
        if (flowgraph->view().has(name)) {
            result.update(Serialize::NodeLayout(flowgraph, name));
        }
    }

    std::vector<std::string> names;
    flowgraph->view().keys(names);
    auto health = PatchHealth(flowgraph, results, before);

    return Json({
        {"status", status},
        {"results", std::move(results)},
        {"block_count", names.size()},
        {"health", std::move(health)},
    });
}

std::string RefHelp(const std::set<std::string>& refs) {
    if (refs.empty()) {
        return " No earlier create_block sets a ref; give one a ref such as 'src' and write '$src', "
               "or use the exact block name.";
    }

    std::string list;
    for (const auto& ref : refs) {
        jst::fmt::format_to(std::back_inserter(list), "{}${}", list.empty() ? "" : ", ", ref);
    }
    return jst::fmt::format(" Refs defined before this operation: {}.", list);
}

// Some servers hand a nested array over as text when they cannot decode it. A complete
// array, or one missing only its closing bracket, is recovered; anything else stays text.
Json DecodeOperations(Json args) {
    if (!args.is_object() || !args.contains("operations") || !args["operations"].is_string()) {
        return args;
    }

    const std::string text = args["operations"].get<std::string>();
    for (const char* closing : {"", "]"}) {
        auto decoded = Json::parse(text + closing, nullptr, false);
        if (decoded.is_array()) {
            args["operations"] = std::move(decoded);
            break;
        }
    }
    return args;
}

std::string ValidateRequest(const Json& args) {
    if (args.contains("operations") && args["operations"].is_string()) {
        return kOperationsText;
    }
    if (!args.contains("operations") || !args["operations"].is_array() || args["operations"].empty()) {
        return "'operations' must be a non-empty array.";
    }
    if (args.contains("flowgraph_name") && !args["flowgraph_name"].is_string()) {
        return "'flowgraph_name' must be a string.";
    }
    for (const auto& [key, value] : args.items()) {
        if (key != "operations" && key != "flowgraph_name") {
            return jst::fmt::format("Unknown patch field '{}'.", key);
        }
    }

    std::set<std::string> refs;
    for (U64 index = 0; index < args["operations"].size(); ++index) {
        const auto& operation = args["operations"][index];
        const std::string prefix = jst::fmt::format("Operation {}", index);

        if (!operation.is_object() || !operation.contains("op") || !operation["op"].is_string()) {
            return jst::fmt::format("{} requires a string 'op'.", prefix);
        }

        const std::string op = operation["op"].get<std::string>();
        const auto* entry = FindOperation(op);
        if (!entry) {
            return jst::fmt::format("{} has unknown op '{}'.", prefix, op);
        }
        if (const auto error = ValidateOperation(*entry, operation); !error.empty()) {
            return jst::fmt::format("{} {}", prefix, error);
        }

        if (operation.contains("ref")) {
            const auto ref = operation["ref"].get<std::string>();
            if (!ref.empty() && !refs.insert(ref).second) {
                return jst::fmt::format("{} repeats ref '{}'.", prefix, ref);
            }
        }
        if (op == "create_block") {
            continue;
        }

        for (const char* key : kRefFields) {
            const std::string value = operation.value(key, std::string{});
            if (IsRef(value) && !refs.contains(RefLabel(refs, value))) {
                return jst::fmt::format("{} field '{}' uses ref '{}', which no earlier create_block in "
                                        "this patch defines.{} Nothing was changed.",
                                        prefix, key, value, RefHelp(refs));
            }
        }
    }

    return {};
}

Json PatchText(const Graph& flowgraph, const Json& args) {
    for (const char* key : {"name", "field", "patch"}) {
        if (!args.contains(key) || !args[key].is_string()) {
            return Response::Error(jst::fmt::format("'{}' must be a string.", key));
        }
    }

    const auto before = IssueKeys(flowgraph);
    Json result = Json::object();
    EditText(flowgraph, args["name"].get<std::string>(), args["field"].get<std::string>(),
             args["patch"].get<std::string>(), result);
    if (result.contains("error")) {
        return result;
    }

    result.update(Serialize::NodeLayout(flowgraph, result["name"].get<std::string>()));
    result["health"] = PatchHealth(flowgraph, Json::array({result}), before);
    return result;
}

Json ApplyPatch(const Graph& flowgraph, const Json& raw) {
    const Json args = DecodeOperations(raw);
    const std::string validationError = ValidateRequest(args);
    if (!validationError.empty()) {
        return Response::Error(validationError);
    }
    if (!flowgraph) {
        return Response::Error("No flowgraph is selected and automatic creation was unavailable.");
    }

    const auto before = IssueKeys(flowgraph);
    Patch patch{flowgraph, {}};
    Json results = Json::array();

    for (const auto& operation : args["operations"]) {
        const std::string op = operation["op"].get<std::string>();
        Json result = {{"index", results.size()}, {"op", op}};

        try {
            FindOperation(op)->apply(patch, operation, result);
        } catch (const std::exception& error) {
            result["error"] = error.what();
            result["may_have_applied"] = true;
        } catch (...) {
            result["error"] = "Operation failed unexpectedly.";
            result["may_have_applied"] = true;
        }

        const bool failed = result.contains("error");
        results.push_back(std::move(result));
        if (failed) {
            return PatchResponse(flowgraph, std::move(results), "partial", before);
        }
    }

    return PatchResponse(flowgraph, std::move(results), "success", before);
}

void AnnotateSize(const Json& operation, const Layout& layout, Json& result) {
    if (operation.value("op", std::string{}) != "set_node_layout" || !result.contains("name")) {
        return;
    }

    const auto found = layout.nodes.find(result["name"].get<std::string>());
    if (found == layout.nodes.end()) {
        return;
    }

    const auto& size = found->second.size;
    const auto dropped = [&](const char* key, F32 applied) {
        if (!operation.contains(key) || !operation[key].is_number()) {
            return false;
        }
        const F32 requested = operation[key].get<F32>();
        return requested > 0.0f && std::abs(applied - requested) > kAppliedTolerance;
    };

    Json missing = Json::array();
    if (dropped("width", size.x)) {
        missing.push_back("width");
    }
    if (dropped("height", size.y)) {
        missing.push_back("height");
    }
    if (!missing.empty()) {
        result["size_not_applied"] = std::move(missing);
    }
}

void AnnotatePatch(Json& body, const Json& raw, const Layout& layout) {
    const Json args = DecodeOperations(raw);
    Geometry::Viewport(body, layout);

    if (!body.contains("results") || !body["results"].is_array()) {
        return;
    }
    Geometry::Overlaps(body, layout);

    if (!args.is_object() || !args.contains("operations") || !args["operations"].is_array()) {
        return;
    }

    const auto& operations = args["operations"];
    auto& results = body["results"];
    for (U64 i = 0; i < operations.size() && i < results.size(); ++i) {
        AnnotateSize(operations[i], layout, results[i]);
    }
}

Entry::NewGraph PatchGraph(const Json& raw) {
    const Json args = DecodeOperations(raw);
    if (!args.is_object() || !ValidateRequest(args).empty()) {
        return {};
    }

    const auto name = args.value("flowgraph_name", std::string("flowgraph"));
    if (name.empty()) {
        return {"flowgraph", {}};
    }
    return {name, {}};
}

}  // namespace

std::vector<Entry> Toolbox::Patch() {
    auto entry = Entry::FromJson({
        "apply_flowgraph_patch",
        "Apply an ordered edit to the selected graph. Only when none is selected, "
        "a graph is created automatically; flowgraph_name names that new graph and "
        "is ignored otherwise. Operations execute sequentially and stop on failure "
        "without rolling back earlier changes, returning status='partial'. A failed "
        "result with may_have_applied=true may have changed the graph before failing, "
        "so inspect it before retrying. "
        "create_block may set ref to a label; later operations in the same call "
        "name that block as '$' plus the label, so a block created with ref 'src' "
        "is written '$src'. Without a ref, use the exact block name. A ref that no "
        "earlier create_block defines rejects the whole call before anything runs. "
        "Refs do not persist across calls. Config is a JSON object. "
        "create_block accepts optional x and y graph coordinates for initial "
        "placement with automatic sizing. Its result includes effective config "
        "with defaults and runtime-defined ports; wiring may require a later patch. "
        "reconfigure_block results echo only the supplied config fields. Both "
        "config readbacks shorten long strings and arrays; inspect_block returns "
        "complete values. "
        "Partial edits of a text config field, such as python code, use the separate "
        "patch_block_text tool; reconfigure_block replaces a whole field. "
        "set_node_layout changes x/y position and/or saved content width/height; "
        "omitted fields stay unchanged. Set only x and y to move without resizing. "
        "Returns attempted operation results, block_count, and health. Each result "
        "describes its operation, not the final whole graph. health.issue_count "
        "covers all issues, while health.issues shows only touched-block and newly "
        "introduced issues; validate_flowgraph lists every issue. Node results "
        "include saved settings and measured outer bounds after layout. "
        "layout_status='pending' has bounds=null; inspect_block or inspect_flowgraph "
        "can retrieve measurements later. Automatic sizing omits saved width and "
        "height. viewport is included when the graph is on screen, with the same "
        "fields and graph coordinates as inspect_flowgraph. Results list overlaps, "
        "the pairs of nodes whose settled bounds intersect, when there are any. "
        "Text and code nodes report content_size, the inner width and height that "
        "show all of their content, and clipped=true when the node is smaller. A "
        "set_node_layout size that did not take effect is listed in "
        "size_not_applied.",
        PatchSchema(),
        {.needsGraph = true, .reportsLayout = true, .mutates = true, .destructive = true},
    }, ApplyPatch);

    entry.newGraph = PatchGraph;
    entry.annotate = AnnotatePatch;

    auto text = Entry::FromJson({
        "patch_block_text",
        "Edit part of one text config field of an existing block, such as python code, "
        "without resending the whole field. Read the current lines with read_block_text "
        "first. patch is plain text made of one or more hunks in text order. Each hunk "
        "starts with a line containing only @@. The lines below it copy a run of "
        "consecutive lines from the field: start each unchanged line with one space, each "
        "line to remove with -, and each line to add with +, then the line exactly as it "
        "is in the field. No line numbers. Keep about three unchanged lines around each "
        "change so the hunk matches exactly one place; a hunk with only + lines appends "
        "to the end. Example patch that changes one line:\n"
        "@@\n"
        " def scale(values):\n"
        "-    factor = 2\n"
        "+    factor = 3\n"
        "     return values * factor\n"
        "All hunks apply or none do. Returns the changed lines rather than the whole "
        "text, line_count, the block state with any diagnostic, health, and the node "
        "layout fields described for apply_flowgraph_patch. To replace a whole field, "
        "use reconfigure_block in apply_flowgraph_patch.",
        R"({"type":"object","required":["name","field","patch"],"properties":{"name":{"type":"string","description":"Exact block name from inspect_flowgraph."},"field":{"type":"string","description":"Text config field, such as code."},"patch":{"type":"string","description":"Plain-text hunks of @@, space, -, and + lines. Copy code characters exactly as read_block_text returns them, adding no backslashes or quotes."}},"additionalProperties":false})",
        {.needsGraph = true, .reportsLayout = true, .mutates = true, .destructive = true},
    }, PatchText);

    return {std::move(entry), std::move(text)};
}

}  // namespace Jetstream::Agent::Detail
