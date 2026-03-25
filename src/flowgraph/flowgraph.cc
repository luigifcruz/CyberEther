#include "jetstream/flowgraph.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/registry.hh"
#include "rapidyaml.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <queue>
#include <ranges>
#include <regex>
#include <string_view>
#include <unordered_set>

namespace Jetstream {

namespace {

struct OutputRef {
    std::string block;
    std::string port;
};

struct BlockState {
    std::string name;
    std::string type;
    DeviceType device;
    RuntimeType runtime;
    ProviderType provider;
    Parser::Map config;
    TensorMap inputs;
};

using OutputLookup = std::unordered_map<U64, OutputRef>;

std::string NormalizeScalar(const std::string& value) {
    if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

OutputLookup BuildOutputLookup(const std::vector<std::string>& blockNames,
                               const std::unordered_map<std::string, std::shared_ptr<Block>>& blocks) {
    OutputLookup lookup;

    for (const auto& name : blockNames) {
        const auto& block = blocks.at(name);
        for (const auto& [port, link] : block->outputs()) {
            const U64 tensorId = link.tensor.id();

            if (tensorId == 0) {
                continue;
            }

            lookup[tensorId] = {name, port};
        }
    }

    return lookup;
}

std::vector<std::string> CollectTensorTypes(const TensorMap& map) {
    std::vector<std::string> types;
    types.reserve(map.size());

    for (const auto& [_, link] : map) {
        const auto& typeName = DataTypeToName(link.tensor.dtype());

        if (!typeName.empty()) {
            types.emplace_back(typeName);
        }
    }

    std::sort(types.begin(), types.end());
    types.erase(std::unique(types.begin(), types.end()), types.end());

    return types;
}

Result SerializeConfigToYaml(const Parser::Map& configMap, ryml::NodeRef& node) {
    node |= ryml::MAP;

    for (const auto& [key, value] : configMap) {
        if (key.empty()) {
            continue;
        }

        if (value.type() == typeid(Parser::Map)) {
            auto child = node.append_child();
            child << ryml::key(key);
            JST_CHECK(SerializeConfigToYaml(std::any_cast<const Parser::Map&>(value), child));
            continue;
        }

        std::string encoded;
        JST_CHECK(Parser::TypedToString(value, encoded));

        if (encoded.empty()) {
            continue;
        }

        auto child = node.append_child();
        child << ryml::key(key);
        child << encoded;

        if (std::count(encoded.begin(), encoded.end(), '\n')) {
            child |= ryml::_WIP_VAL_LITERAL;
        }
    }

    return Result::SUCCESS;
}

Result DeserializeConfigFromYaml(ryml::ConstNodeRef node, Parser::Map& configMap) {
    if (!node.is_map()) {
        JST_ERROR("[FLOWGRAPH] Config node must be a map.");
        return Result::ERROR;
    }

    Parser::Map decoded;

    for (const auto& element : node.children()) {
        std::string key(element.key().str, element.key().len);

        if (element.is_map()) {
            Parser::Map nested;
            JST_CHECK(DeserializeConfigFromYaml(element, nested));
            decoded[key] = std::move(nested);
            continue;
        }

        std::string value;
        if (!element.has_val()) {
            JST_ERROR("[FLOWGRAPH] Config '{}' must be a scalar or map.", key);
            return Result::ERROR;
        }

        element >> value;
        decoded[key] = NormalizeScalar(value);
    }

    configMap = std::move(decoded);
    return Result::SUCCESS;
}

void SetScalarNode(ryml::NodeRef& node, const char* key, const std::string& value, bool literalOnNewline = false) {
    if (value.empty()) {
        return;
    }

    auto child = node.append_child();
    child << ryml::key(key) << value;

    if (literalOnNewline && std::count(value.begin(), value.end(), '\n')) {
        child |= ryml::_WIP_VAL_LITERAL;
    }
}

Result SerializeMetaToYaml(const std::unordered_map<std::string, Flowgraph::Meta>& meta, ryml::NodeRef& node) {
    node |= ryml::MAP;

    for (const auto& [key, data] : meta) {
        if (key.empty() || data.empty()) {
            continue;
        }

        auto entryNode = node.append_child();
        entryNode << ryml::key(key);
        JST_CHECK(SerializeConfigToYaml(data, entryNode));
    }

    return Result::SUCCESS;
}

Result DeserializeMetaFromYaml(ryml::ConstNodeRef node, std::unordered_map<std::string, Flowgraph::Meta>& meta) {
    if (!node.is_map()) {
        return Result::ERROR;
    }

    for (const auto& child : node.children()) {
        std::string key(child.key().str, child.key().len);

        if (!child.is_map()) {
            JST_ERROR("[FLOWGRAPH] Meta entry '{}' must be a map.", key);
            return Result::ERROR;
        }

        Flowgraph::Meta data;
        JST_CHECK(DeserializeConfigFromYaml(child, data));

        meta[key] = std::move(data);
    }

    return Result::SUCCESS;
}

std::optional<OutputRef> ParseGraphReference(const std::string& value) {
    static const std::regex pattern(R"(\$\{graph\.([^.]+)\.output\.([^.]+)\})");
    std::smatch matches;
    if (!std::regex_match(value, matches, pattern) || matches.size() != 3) {
        return std::nullopt;
    }

    return OutputRef{matches[1].str(), matches[2].str()};
}

Result ReadScalar(const ryml::NodeRef& node, const char* key, std::string& value) {
    const auto k = ryml::to_csubstr(key);

    value.clear();

    if (!node.has_child(k)) {
        return Result::SUCCESS;
    }

    auto child = node[k];
    if (!child.has_val()) {
        JST_ERROR("[FLOWGRAPH] Entry '{}' must be a scalar value.", key);
        return Result::ERROR;
    }

    child >> value;
    value = NormalizeScalar(value);
    return Result::SUCCESS;
}

}  // namespace

struct Flowgraph::Impl {
    bool created = false;

    std::shared_ptr<Instance> instance;
    std::shared_ptr<Render::Window> render;
    std::shared_ptr<Compositor> compositor;

    std::shared_ptr<Scheduler> scheduler;
    std::unordered_map<std::string, std::shared_ptr<Block>> blocks;
    std::vector<std::string> blockOrder;
    std::unordered_map<std::string, std::vector<std::string>> edges;

    std::string title;
    std::string summary;
    std::string author;
    std::string license;
    std::string description;
    std::string path;

    std::unordered_map<std::string, Flowgraph::Meta> meta;
    std::unordered_map<std::string, std::unordered_map<std::string, Flowgraph::Meta>> blockMeta;

    Result resolveInputs(const TensorMap& requested, TensorMap& resolved) const;
    std::vector<std::string> collectDownstream(const std::string& name) const;
};

Result Flowgraph::Impl::resolveInputs(const TensorMap& requested, TensorMap& resolved) const {
    Result result = Result::SUCCESS;

    for (const auto& [slot, link] : requested) {
        if (!blocks.contains(link.block)) {
            JST_ERROR("[FLOWGRAPH] Block '{}' does not exist.", link.block);
            return Result::ERROR;
        }

        const auto& outputs = blocks.at(link.block)->outputs();

        if (!outputs.contains(link.port)) {
            JST_WARN("[FLOWGRAPH] Block '{}' has no output '{}' to satisfy connection '{}'.", link.block,
                                                                                              link.port,
                                                                                              slot);
            resolved.emplace(slot, link);
            result = Result::INCOMPLETE;
            continue;
        }

        resolved.emplace(slot, outputs.at(link.port));
    }

    return result;
}

std::vector<std::string> Flowgraph::Impl::collectDownstream(const std::string& name) const {
    std::vector<std::string> result;
    std::queue<std::string> queue;
    std::unordered_set<std::string> visited;

    if (edges.contains(name)) {
        for (const auto& dep : edges.at(name)) {
            queue.push(dep);
        }
    }

    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop();

        if (visited.contains(current)) {
            continue;
        }

        visited.insert(current);
        result.push_back(current);

        if (edges.contains(current)) {
            for (const auto& dep : edges.at(current)) {
                if (!visited.contains(dep)) {
                    queue.push(dep);
                }
            }
        }
    }

    return result;
}

Flowgraph::Flowgraph() {
    impl = std::make_unique<Impl>();
}

Flowgraph::~Flowgraph() {
    impl.reset();
}

Result Flowgraph::create(const Config& config,
                         const std::shared_ptr<Instance>& instance,
                         const std::shared_ptr<Render::Window>& render,
                         const std::shared_ptr<Compositor>& compositor) {
    JST_INFO("[FLOWGRAPH] Creating flowgraph.");
    JST_ASSERT(!impl->created, "[FLOWGRAPH] Flowgraph already created.");

    // Set implementation variables.

    impl->instance = instance;
    impl->render = render;
    impl->compositor = compositor;

    // Build Scheduler.

    impl->scheduler = std::make_shared<Scheduler>(config.scheduler);

    // Create Scheduler.

    JST_CHECK(impl->scheduler->create(impl->instance));

    impl->created = true;
    return Result::SUCCESS;
}

Result Flowgraph::destroy() {
    JST_INFO("[FLOWGRAPH] Destroying flowgraph.");
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.");

    if (!impl->blocks.empty()) {
        JST_WARN("[FLOWGRAPH] Flowgraph still has {} blocks for destruction.", impl->blocks.size());

        // Destroy blocks in reverse order of creation.

        for (const auto& name : std::ranges::reverse_view(impl->blockOrder)) {
            if (impl->blocks.contains(name)) {
                JST_CHECK(impl->blocks.at(name)->destroy());
            }
        }
    }

    if (impl->scheduler) {
        JST_CHECK(impl->scheduler->destroy());
    }

    impl->blocks.clear();
    impl->blockOrder.clear();
    impl->edges.clear();
    impl->path.clear();

    impl->created = false;
    return Result::SUCCESS;
}

Result Flowgraph::blockCreate(const std::string name,
                              const Block::Config& config,
                              const TensorMap& inputs,
                              const DeviceType& device,
                              const RuntimeType& runtime,
                              const ProviderType& provider) {
    Parser::Map serializedConfig;
    JST_CHECK(config.serialize(serializedConfig));
    JST_CHECK(this->blockCreate(name, config.type(), serializedConfig, inputs, device, runtime, provider));

    return Result::SUCCESS;
}

Result Flowgraph::blockCreate(const std::string name,
                              const std::string type,
                              const Parser::Map& config,
                              const TensorMap& inputs,
                              const DeviceType& device,
                              const RuntimeType& runtime,
                              const ProviderType& provider) {
    JST_INFO("[FLOWGRAPH] Creating block '{}' of type '{}'.", name, type);
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.");

    if (impl->blocks.contains(name)) {
        JST_ERROR("[FLOWGRAPH] Create: Block '{}' already exist.", name);
        return Result::ERROR;
    }

    std::shared_ptr<Block> block;
    JST_CHECK(Registry::BuildBlock(type, block));

    TensorMap resolvedInputs;
    JST_CHECK_ALLOW(impl->resolveInputs(inputs, resolvedInputs), Result::INCOMPLETE);

    impl->blocks[name] = block;
    impl->blockOrder.push_back(name);

    const auto result = block->create(name,
                                      device,
                                      runtime,
                                      provider,
                                      config,
                                      resolvedInputs,
                                      impl->instance,
                                      impl->render,
                                      impl->scheduler);

    if (result != Result::SUCCESS && result != Result::INCOMPLETE && result != Result::ERROR) {
        impl->blocks.erase(name);
        impl->blockOrder.erase(std::remove(impl->blockOrder.begin(), impl->blockOrder.end(), name),
                               impl->blockOrder.end());
        return result;
    }

    // Register dependency list.

    for (const auto& [_, link] : inputs) {
        impl->edges[link.block].push_back(name);
    }

    return Result::SUCCESS;
}

Result Flowgraph::blockDestroy(const std::string name, bool propagate) {
    JST_INFO("[FLOWGRAPH] Destroying block '{}'.", name);
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.");

    if (!impl->blocks.contains(name)) {
        JST_ERROR("[FLOWGRAPH] Cannot destroy block '{}' because it doesn't exist.", name);
        return Result::ERROR;
    }

    // Handle downstream blocks that depend on this block.

    if (propagate and impl->edges.contains(name)) {
        // 1. Collect all downstream blocks (transitive closure).

        std::vector<BlockState> downstreamStates;

        for (const auto& depName : impl->collectDownstream(name)) {
            const auto& dep = impl->blocks.at(depName);

            BlockState state;
            state.name = depName;
            state.type = dep->config().type();
            state.device = dep->device();
            state.runtime = dep->runtime();
            state.provider = dep->provider();
            JST_CHECK(dep->config().serialize(state.config));

            for (const auto& [slot, link] : dep->inputs()) {
                if (link.block != name) {
                    state.inputs[slot] = {link.block, link.port, {}};
                }
            }

            downstreamStates.push_back(std::move(state));
        }

        // 2. Destroy downstream blocks in reverse order.

        for (const auto& state : std::ranges::reverse_view(downstreamStates)) {
            JST_CHECK(blockDestroy(state.name, false));
        }

        // 3. Destroy the target block.

        JST_CHECK(blockDestroy(name, false));

        // 4. Recreate downstream blocks in forward order (with severed connections).

        for (const auto& state : downstreamStates) {
            JST_CHECK_ALLOW(blockCreate(state.name, state.type, state.config,
                                        state.inputs, state.device, state.runtime, state.provider),
                            Result::INCOMPLETE);
        }

        return Result::SUCCESS;
    }

    if (impl->edges.contains(name)) {
        impl->edges.erase(name);
    }

    // Remove dependency list.

    for (auto& [_, deps] : impl->edges) {
        deps.erase(std::remove(deps.begin(), deps.end(), name), deps.end());
    }

    // Destroy block and remove it from state.

    JST_CHECK(impl->blocks.at(name)->destroy());
    impl->blocks.erase(name);
    impl->blockOrder.erase(std::remove(impl->blockOrder.begin(), impl->blockOrder.end(), name),
                           impl->blockOrder.end());

    return Result::SUCCESS;
}

Result Flowgraph::blockConnect(const std::string blockName,
                               const std::string inputPort,
                               const std::string sourceBlock,
                               const std::string sourcePort) {
    JST_INFO("[FLOWGRAPH] Connecting '{}.{}' to '{}.{}'.",
             sourceBlock, sourcePort, blockName, inputPort);
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.");

    if (!impl->blocks.contains(blockName)) {
        JST_ERROR("[FLOWGRAPH] Block '{}' doesn't exist.", blockName);
        return Result::ERROR;
    }

    if (!impl->blocks.contains(sourceBlock)) {
        JST_ERROR("[FLOWGRAPH] Source block '{}' doesn't exist.", sourceBlock);
        return Result::ERROR;
    }

    const auto& block = impl->blocks.at(blockName);
    const auto& source = impl->blocks.at(sourceBlock);

    if (!source->outputs().contains(sourcePort)) {
        JST_WARN("[FLOWGRAPH] Source port '{}.{}' doesn't exist.", sourceBlock, sourcePort);
    }

    // 1. Collect downstream blocks.

    std::vector<BlockState> downstreamStates;

    for (const auto& depName : impl->collectDownstream(blockName)) {
        const auto& dep = impl->blocks.at(depName);

        BlockState state;
        state.name = depName;
        state.type = dep->config().type();
        state.device = dep->device();
        state.runtime = dep->runtime();
        state.provider = dep->provider();
        JST_CHECK(dep->config().serialize(state.config));

        for (const auto& [slot, link] : dep->inputs()) {
            state.inputs[slot] = {link.block, link.port, {}};
        }

        downstreamStates.push_back(std::move(state));
    }

    // 2. Destroy downstream blocks in reverse order.

    for (const auto& state : std::ranges::reverse_view(downstreamStates)) {
        JST_CHECK(blockDestroy(state.name, false));
    }

    // 3. Save target block state and modify inputs.

    const auto type = block->config().type();
    const auto device = block->device();
    const auto runtime = block->runtime();
    const auto provider = block->provider();

    Parser::Map serializedConfig;
    JST_CHECK(block->config().serialize(serializedConfig));

    TensorMap newInputs = block->inputs();
    newInputs[inputPort] = {sourceBlock, sourcePort, {}};

    // 4. Destroy and recreate target block.

    JST_CHECK(this->blockDestroy(blockName, false));
    JST_CHECK_ALLOW(this->blockCreate(blockName, type, serializedConfig, newInputs, device, runtime, provider),
                    Result::INCOMPLETE);

    // 5. Recreate downstream blocks in forward order.

    for (const auto& state : downstreamStates) {
        JST_CHECK_ALLOW(blockCreate(state.name, state.type, state.config,
                                    state.inputs, state.device, state.runtime, state.provider),
                        Result::INCOMPLETE);
    }

    return Result::SUCCESS;
}

Result Flowgraph::blockDisconnect(const std::string blockName,
                                  const std::string inputPort) {
    JST_INFO("[FLOWGRAPH] Disconnecting '{}.{}'.", blockName, inputPort);
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.");

    if (!impl->blocks.contains(blockName)) {
        JST_ERROR("[FLOWGRAPH] Block '{}' doesn't exist.", blockName);
        return Result::ERROR;
    }

    const auto& block = impl->blocks.at(blockName);

    TensorMap newInputs = block->inputs();
    if (!newInputs.contains(inputPort)) {
        JST_ERROR("[FLOWGRAPH] Input port '{}.{}' is not connected.", blockName, inputPort);
        return Result::ERROR;
    }

    // 1. Collect downstream blocks.

    std::vector<BlockState> downstreamStates;

    for (const auto& depName : impl->collectDownstream(blockName)) {
        const auto& dep = impl->blocks.at(depName);

        BlockState state;
        state.name = depName;
        state.type = dep->config().type();
        state.device = dep->device();
        state.runtime = dep->runtime();
        state.provider = dep->provider();
        JST_CHECK(dep->config().serialize(state.config));

        for (const auto& [slot, link] : dep->inputs()) {
            state.inputs[slot] = {link.block, link.port, {}};
        }

        downstreamStates.push_back(std::move(state));
    }

    // 2. Destroy downstream blocks in reverse order.

    for (const auto& state : std::ranges::reverse_view(downstreamStates)) {
        JST_CHECK(blockDestroy(state.name, false));
    }

    // 3. Save target block state and modify inputs.

    const auto type = block->config().type();
    const auto device = block->device();
    const auto runtime = block->runtime();
    const auto provider = block->provider();

    Parser::Map serializedConfig;
    JST_CHECK(block->config().serialize(serializedConfig));

    newInputs.erase(inputPort);

    // 4. Destroy and recreate target block.

    JST_CHECK(this->blockDestroy(blockName, false));
    JST_CHECK_ALLOW(this->blockCreate(blockName, type, serializedConfig, newInputs, device, runtime, provider),
                    Result::INCOMPLETE);

    // 5. Recreate downstream blocks in forward order.

    for (const auto& state : downstreamStates) {
        JST_CHECK_ALLOW(blockCreate(state.name, state.type, state.config,
                                    state.inputs, state.device, state.runtime, state.provider),
                        Result::INCOMPLETE);
    }

    return Result::SUCCESS;
}

const std::unordered_map<std::string, std::shared_ptr<Block>>& Flowgraph::blockList() const {
    return impl->blocks;
}

Result Flowgraph::blockReconfigure(const std::string name, const Parser::Map& config) {
    if (!impl->blocks.contains(name)) {
        JST_ERROR("[FLOWGRAPH] Cannot update block '{}' because it doesn't exist.", name);
        return Result::ERROR;
    }

    const auto result = impl->blocks.at(name)->reconfigure(config);

    if (result == Result::RECREATE) {
        JST_INFO("[FLOWGRAPH] Block '{}' requested recreation.", name);
        return blockRecreate(name, config);
    }

    JST_CHECK(result);

    return Result::SUCCESS;
}

Result Flowgraph::blockRecreate(const std::string name, const Parser::Map& config) {
    JST_INFO("[FLOWGRAPH] Recreating block '{}' and downstream blocks.", name);
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.");

    if (!impl->blocks.contains(name)) {
        JST_ERROR("[FLOWGRAPH] Cannot recreate block '{}' because it doesn't exist.", name);
        return Result::ERROR;
    }

    // 1. Collect state of target block and all downstream blocks.

    std::vector<BlockState> blocksToRecreate;

    // Add target block with new config.

    {
        const auto& block = impl->blocks.at(name);

        BlockState state;
        state.name = name;
        state.type = block->config().type();
        state.device = block->device();
        state.runtime = block->runtime();
        state.provider = block->provider();
        state.config = config;

        for (const auto& [slot, link] : block->inputs()) {
            state.inputs[slot] = {link.block, link.port, {}};
        }

        blocksToRecreate.push_back(std::move(state));
    }

    // Add downstream blocks with their current config.

    for (const auto& depName : impl->collectDownstream(name)) {
        const auto& dep = impl->blocks.at(depName);

        BlockState state;
        state.name = depName;
        state.type = dep->config().type();
        state.device = dep->device();
        state.runtime = dep->runtime();
        state.provider = dep->provider();
        JST_CHECK(dep->config().serialize(state.config));

        for (const auto& [slot, link] : dep->inputs()) {
            state.inputs[slot] = {link.block, link.port, {}};
        }

        blocksToRecreate.push_back(std::move(state));
    }

    // 2. Destroy all blocks in reverse order.

    for (const auto& state : std::ranges::reverse_view(blocksToRecreate)) {
        JST_CHECK(blockDestroy(state.name, false));
    }

    // 3. Recreate all blocks in forward order.

    for (const auto& state : blocksToRecreate) {
        JST_CHECK_ALLOW(blockCreate(state.name, state.type, state.config,
                                    state.inputs, state.device, state.runtime, state.provider),
                        Result::INCOMPLETE);
    }

    return Result::SUCCESS;
}

Result Flowgraph::blockConfig(const std::string name, Parser::Map& config) const {
    if (!impl->blocks.contains(name)) {
        JST_ERROR("[FLOWGRAPH] Cannot get block '{}' configuration because it doesn't exist.", name);
        return Result::ERROR;
    }

    JST_CHECK(impl->blocks.at(name)->config(config));

    return Result::SUCCESS;
}

Result Flowgraph::importFromFile(const std::string& path) {
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.");
    impl->path = path;

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        JST_ERROR("[FLOWGRAPH] Can't open flowgraph file '{}'.", path);
        return Result::ERROR;
    }

    std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    return importFromBlob(blob);
}

Result Flowgraph::importFromBlob(const std::vector<char>& blob) {
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.")

    std::string yamlText(blob.begin(), blob.end());
    ryml::Tree tree;
    try {
        tree = ryml::parse_in_arena(ryml::to_csubstr(yamlText));
    } catch (...) {
        JST_ERROR("[FLOWGRAPH] Failed to parse flowgraph blob.");
        return Result::ERROR;
    }

    auto root = tree.rootref();
    if (!root.is_map() && root.num_children() > 0) {
        root = root[0];
    }

    if (!root.is_map()) {
        JST_ERROR("[FLOWGRAPH] Flowgraph root is not a map.");
        return Result::ERROR;
    }

    std::string version;
    JST_CHECK(ReadScalar(root, "version", version));
    if (version.empty()) {
        JST_CHECK(ReadScalar(root, "protocolVersion", version));
    }
    if (!version.empty() && version != "1.0.0") {
        JST_ERROR("[FLOWGRAPH] Invalid flowgraph version '{}'.", version);
        return Result::ERROR;
    }

    std::string title;
    std::string summary;
    std::string author;
    std::string license;
    std::string description;

    JST_CHECK(ReadScalar(root, "title", title));
    JST_CHECK(ReadScalar(root, "summary", summary));
    JST_CHECK(ReadScalar(root, "author", author));
    JST_CHECK(ReadScalar(root, "license", license));
    JST_CHECK(ReadScalar(root, "description", description));

    if (!title.empty()) {
        impl->title = title;
    }
    if (!summary.empty()) {
        impl->summary = summary;
    }
    if (!author.empty()) {
        impl->author = author;
    }
    if (!license.empty()) {
        impl->license = license;
    }
    if (!description.empty()) {
        impl->description = description;
    }

    if (root.has_child("meta")) {
        auto metaNode = root["meta"];
        JST_CHECK(DeserializeMetaFromYaml(metaNode, impl->meta));
    }

    if (!root.has_child("graph")) {
        return Result::SUCCESS;
    }

    auto graph = root["graph"];
    if (!graph.is_map()) {
        JST_ERROR("[FLOWGRAPH] Graph entry is not a map.");
        return Result::ERROR;
    }

    struct NodeDef {
        std::string type;
        DeviceType device = DeviceType::CPU;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider = "generic";
        Parser::Map config;
        std::unordered_map<std::string, OutputRef> inputs;
        std::unordered_map<std::string, Flowgraph::Meta> meta;
    };

    std::unordered_map<std::string, NodeDef> nodes;

    for (const auto& blockNode : graph.children()) {
        if (!blockNode.has_key()) {
            JST_ERROR("[FLOWGRAPH] Block without a name found.");
            return Result::ERROR;
        }

        std::string name(blockNode.key().str, blockNode.key().len);
        NodeDef def;

        if (!blockNode.has_child("module")) {
            JST_ERROR("[FLOWGRAPH] Block '{}' is missing 'module'.", name);
            return Result::ERROR;
        }
        auto moduleNode = blockNode["module"];
        if (!moduleNode.has_val()) {
            JST_ERROR("[FLOWGRAPH] Block '{}' has non-scalar 'module'.", name);
            return Result::ERROR;
        }
        moduleNode >> def.type;

        if (blockNode.has_child("device")) {
            std::string deviceStr;
            auto deviceNode = blockNode["device"];
            if (!deviceNode.has_val()) {
                JST_ERROR("[FLOWGRAPH] Block '{}' has non-scalar 'device'.", name);
                return Result::ERROR;
            }
            deviceNode >> deviceStr;
            try {
                def.device = StringToDevice(deviceStr);
            } catch (...) {
                JST_ERROR("[FLOWGRAPH] Block '{}' has invalid device '{}'.", name, deviceStr);
                return Result::ERROR;
            }
        }

        if (blockNode.has_child("runtime")) {
            std::string runtimeStr;
            auto runtimeNode = blockNode["runtime"];
            if (!runtimeNode.has_val()) {
                JST_ERROR("[FLOWGRAPH] Block '{}' has non-scalar 'runtime'.", name);
                return Result::ERROR;
            }
            runtimeNode >> runtimeStr;
            def.runtime = StringToRuntime(runtimeStr);
            if (def.runtime == RuntimeType::NONE) {
                JST_ERROR("[FLOWGRAPH] Block '{}' has invalid runtime '{}'.", name, runtimeStr);
                return Result::ERROR;
            }
        }

        if (blockNode.has_child("provider")) {
            auto providerNode = blockNode["provider"];
            if (!providerNode.has_val()) {
                JST_ERROR("[FLOWGRAPH] Block '{}' has non-scalar 'provider'.", name);
                return Result::ERROR;
            }
            providerNode >> def.provider;
        }

        if (blockNode.has_child("config")) {
            auto cfg = blockNode["config"];
            if (!cfg.is_map()) {
                JST_ERROR("[FLOWGRAPH] Block '{}' config must be a map.", name);
                return Result::ERROR;
            }

            JST_CHECK(DeserializeConfigFromYaml(cfg, def.config));
        }

        if (blockNode.has_child("input")) {
            auto input = blockNode["input"];
            if (!input.is_map()) {
                JST_ERROR("[FLOWGRAPH] Block '{}' input must be a map.", name);
                return Result::ERROR;
            }

            for (const auto& element : input.children()) {
                std::string key(element.key().str, element.key().len);
                std::string value;
                if (!element.has_val()) {
                    JST_ERROR("[FLOWGRAPH] Block '{}' input '{}' must be a scalar.", name, key);
                    return Result::ERROR;
                }
                element >> value;

                const auto ref = ParseGraphReference(value);
                if (!ref.has_value()) {
                    JST_ERROR("[FLOWGRAPH] Block '{}' input '{}' has invalid link '{}'.", name, key, value);
                    return Result::ERROR;
                }

                def.inputs[key] = ref.value();
            }
        }

        if (blockNode.has_child("meta")) {
            auto metaNode = blockNode["meta"];
            JST_CHECK(DeserializeMetaFromYaml(metaNode, def.meta));
        }

        nodes[name] = std::move(def);
    }

    std::unordered_map<std::string, size_t> indegree;
    std::unordered_map<std::string, std::vector<std::string>> dependents;

    for (const auto& [name, def] : nodes) {
        indegree[name] = 0;
    }

    for (const auto& [name, def] : nodes) {
        for (const auto& [_, input] : def.inputs) {
            if (!nodes.contains(input.block) && !impl->blocks.contains(input.block)) {
                JST_ERROR("[FLOWGRAPH] Block '{}' depends on missing block '{}'.", name, input.block);
                return Result::ERROR;
            }

            // Only track dependencies within the YAML blob for topological sort.
            if (nodes.contains(input.block)) {
                indegree[name] += 1;
                dependents[input.block].push_back(name);
            }
        }
    }

    std::queue<std::string> ready;
    for (const auto& [name, degree] : indegree) {
        if (degree == 0) {
            ready.push(name);
        }
    }

    std::vector<std::string> order;
    while (!ready.empty()) {
        const auto current = ready.front();
        ready.pop();
        order.push_back(current);

        for (const auto& next : dependents[current]) {
            if (--indegree[next] == 0) {
                ready.push(next);
            }
        }
    }

    if (order.size() != nodes.size()) {
        JST_ERROR("[FLOWGRAPH] Detected a cycle or unresolved dependency while importing flowgraph.");
        return Result::ERROR;
    }

    for (const auto& name : order) {
        const auto& def = nodes.at(name);

        TensorMap requestedInputs;
        for (const auto& [slot, ref] : def.inputs) {
            requestedInputs[slot] = {ref.block, ref.port, {}};
        }

        if (!def.meta.empty()) {
            impl->blockMeta[name] = def.meta;
        }

        JST_CHECK(blockCreate(name, def.type, def.config, requestedInputs, def.device, def.runtime, def.provider));
    }

    return Result::SUCCESS;
}

Result Flowgraph::exportToFile(const std::string& path) {
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.");
    impl->path = path;

    if (impl->path.empty()) {
        JST_ERROR("[FLOWGRAPH] Filepath is empty.");
        return Result::ERROR;
    }

    JST_INFO("[FLOWGRAPH] Exporting flowgraph to file '{}'.", impl->path);

    std::vector<char> blob;
    JST_CHECK(exportToBlob(blob));

    const auto parent = std::filesystem::path(impl->path).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            JST_ERROR("[FLOWGRAPH] Cannot create directory '{}'.", parent.string());
            return Result::ERROR;
        }
    }

    std::ofstream file(impl->path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
        JST_ERROR("[FLOWGRAPH] Can't open flowgraph file '{}'.", impl->path);
        return Result::ERROR;
    }

    file.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    file.close();

    return Result::SUCCESS;
}

Result Flowgraph::exportToBlob(std::vector<char>& blob) {
    JST_ASSERT(impl->created, "[FLOWGRAPH] Flowgraph not created.");

    ryml::Tree yaml;
    auto root = yaml.rootref();
    root |= ryml::MAP;

    root["version"] << "1.0.0";

    SetScalarNode(root, "title", impl->title);
    SetScalarNode(root, "summary", impl->summary);
    SetScalarNode(root, "author", impl->author);
    SetScalarNode(root, "license", impl->license);
    SetScalarNode(root, "description", impl->description, true);

    if (!impl->blocks.empty()) {
        ryml::NodeRef graph = root["graph"];
        graph |= ryml::MAP;

        const auto outputLookup = BuildOutputLookup(impl->blockOrder, impl->blocks);

        for (const auto& name : impl->blockOrder) {
            const auto& block = impl->blocks.at(name);

            auto blockNode = graph.append_child();
            blockNode << ryml::key(name);
            blockNode |= ryml::MAP;

            blockNode.append_child() << ryml::key("module") << block->config().type();
            blockNode.append_child() << ryml::key("device") << GetDeviceName(block->device());
            blockNode.append_child() << ryml::key("runtime") << GetRuntimeName(block->runtime());
            blockNode.append_child() << ryml::key("provider") << block->provider();

            const auto inputTypes = CollectTensorTypes(block->inputs());
            const auto outputTypes = CollectTensorTypes(block->outputs());

            const std::string inputType = inputTypes.empty() ? "" : inputTypes.front();
            const std::string outputType = outputTypes.empty() ? "" : outputTypes.front();

            if (!inputType.empty() && (outputType.empty() || inputType == outputType)) {
                blockNode.append_child() << ryml::key("dataType") << inputType;
            } else {
                if (!inputType.empty()) {
                    blockNode.append_child() << ryml::key("inputDataType") << inputType;
                }
                if (!outputType.empty()) {
                    blockNode.append_child() << ryml::key("outputDataType") << outputType;
                }
            }

            Parser::Map configMap;
            JST_CHECK(block->config(configMap));
            if (!configMap.empty()) {
                auto configNode = blockNode.append_child();
                configNode << ryml::key("config");
                JST_CHECK(SerializeConfigToYaml(configMap, configNode));
            }

            const auto& inputs = block->inputs();
            if (!inputs.empty()) {
                auto inputNode = blockNode.append_child();
                inputNode << ryml::key("input");
                inputNode |= ryml::MAP;

                for (const auto& [slot, link] : inputs) {
                    const U64 tensorId = link.tensor.id();

                    std::string producerBlock = link.block;
                    std::string producerPort = link.port;

                    if (tensorId != 0 && outputLookup.contains(tensorId)) {
                        producerBlock = outputLookup.at(tensorId).block;
                        producerPort = outputLookup.at(tensorId).port;
                    }

                    if (producerBlock.empty() || producerPort.empty()) {
                        JST_ERROR("[FLOWGRAPH] Cannot resolve connection for input '{}.{}'.", name, slot);
                        return Result::ERROR;
                    }

                    const auto value = jst::fmt::format("${{graph.{}.output.{}}}", producerBlock, producerPort);
                    inputNode.append_child() << ryml::key(slot) << value;
                }
            }

            if (impl->blockMeta.contains(name) && !impl->blockMeta.at(name).empty()) {
                auto metaNode = blockNode.append_child();
                metaNode << ryml::key("meta");
                JST_CHECK(SerializeMetaToYaml(impl->blockMeta.at(name), metaNode));
            }
        }
    }

    if (!impl->meta.empty()) {
        auto metaNode = root.append_child();
        metaNode << ryml::key("meta");
        JST_CHECK(SerializeMetaToYaml(impl->meta, metaNode));
    }

    auto emitted = ryml::emitrs_yaml<std::vector<char>>(yaml);
    std::string yamlText(emitted.begin(), emitted.end());

    const auto versionPos = yamlText.find("version: ");
    if (versionPos != std::string::npos) {
        const auto newlinePos = yamlText.find('\n', versionPos);
        if (newlinePos != std::string::npos && newlinePos + 1 < yamlText.size() && yamlText[newlinePos + 1] != '\n') {
            yamlText.insert(newlinePos + 1, "\n");
        }
    }

    blob.clear();
    const char header[] = {'-', '-', '-', '\n'};
    blob.insert(blob.end(), std::begin(header), std::end(header));
    blob.insert(blob.end(), yamlText.begin(), yamlText.end());

    return Result::SUCCESS;
}

Result Flowgraph::compute() {
    if (!impl->created) {
        return Result::SUCCESS;
    }

    if (impl->scheduler) {
        JST_CHECK(impl->scheduler->compute());
    }

    return Result::SUCCESS;
}

Result Flowgraph::present() {
    if (!impl->created) {
        return Result::SUCCESS;
    }

    if (impl->scheduler) {
        JST_CHECK(impl->scheduler->present());
    }

    return Result::SUCCESS;
}

const std::unordered_map<std::string, std::shared_ptr<Runtime::Metrics>>& Flowgraph::metrics() const {
    return impl->scheduler->metrics();
}

const std::string& Flowgraph::title() const {
    return impl->title;
}

const std::string& Flowgraph::summary() const {
    return impl->summary;
}

const std::string& Flowgraph::author() const {
    return impl->author;
}

const std::string& Flowgraph::license() const {
    return impl->license;
}

const std::string& Flowgraph::description() const {
    return impl->description;
}

const std::string& Flowgraph::path() const {
    return impl->path;
}

Result Flowgraph::setTitle(const std::string& title) {
    impl->title = title;
    return Result::SUCCESS;
}

Result Flowgraph::setSummary(const std::string& summary) {
    impl->summary = summary;
    return Result::SUCCESS;
}

Result Flowgraph::setAuthor(const std::string& author) {
    impl->author = author;
    return Result::SUCCESS;
}

Result Flowgraph::setLicense(const std::string& license) {
    impl->license = license;
    return Result::SUCCESS;
}

Result Flowgraph::setDescription(const std::string& description) {
    impl->description = description;
    return Result::SUCCESS;
}

Result Flowgraph::getMeta(const std::string& key, Meta& data, const std::string& block) const {
    if (block.empty()) {
        if (impl->meta.contains(key)) {
            data = impl->meta.at(key);
        }
    } else if (impl->blockMeta.contains(block) && impl->blockMeta.at(block).contains(key)) {
        data = impl->blockMeta.at(block).at(key);
    }
    return Result::SUCCESS;
}

Result Flowgraph::setMeta(const std::string& key, const Meta& data, const std::string& block) {
    if (block.empty()) {
        impl->meta[key] = data;
    } else {
        impl->blockMeta[block][key] = data;
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream
