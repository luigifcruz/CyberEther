#include "jetstream/flowgraph.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/registry.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <queue>
#include <ranges>
#include <regex>
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

struct FlowgraphBlockDocument {
    std::string name;
    std::string module;
    DeviceType device = DeviceType::CPU;
    RuntimeType runtime = RuntimeType::NATIVE;
    ProviderType provider = "generic";
    std::optional<Parser::Map> config;
    std::optional<Parser::Map> input;
    std::optional<Parser::Map> meta;

    JST_SERDES(name, module, device, runtime, provider, config, input, meta);
};

struct FlowgraphDocument {
    std::string version = "2";
    std::optional<std::string> title;
    std::optional<std::string> summary;
    std::optional<std::string> author;
    std::optional<std::string> license;
    std::optional<std::string> description;
    std::vector<FlowgraphBlockDocument> graph;
    std::optional<Parser::Map> meta;

    JST_SERDES(version, title, summary, author, license, description, graph, meta);
};

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

std::optional<Parser::Map> CompactMetaMap(const Parser::Map& meta) {
    Parser::Map compact;

    for (const auto& entry : meta) {
        if (entry.value.type() != typeid(Parser::Map)) {
            continue;
        }

        const auto& value = std::any_cast<const Parser::Map&>(entry.value);
        if (!value.empty()) {
            compact[entry.key] = value;
        }
    }

    if (compact.empty()) {
        return std::nullopt;
    }

    return compact;
}

Result MigrateFlowgraphVersion100To200(Parser::Map& root) {
    if (!root.contains("version") && root.contains("protocolVersion")) {
        root["version"] = root.at("protocolVersion");
    }

    std::string version;
    JST_CHECK(Parser::Deserialize(root, "version", version));

    if (version != "1.0.0") {
        return Result::SUCCESS;
    }

    if (root.contains("graph") && root.at("graph").type() == typeid(Parser::Map)) {
        const auto& legacyGraph = std::any_cast<const Parser::Map&>(root.at("graph"));
        Parser::Sequence graph;
        graph.reserve(legacyGraph.size());

        for (const auto& [name, encodedBlock] : legacyGraph) {
            if (encodedBlock.type() != typeid(Parser::Map)) {
                JST_ERROR("[FLOWGRAPH] Block '{}' must serialize to a map.", name);
                return Result::ERROR;
            }

            Parser::Map block = std::any_cast<const Parser::Map&>(encodedBlock);

            block["name"] = name;
            graph.push_back(std::move(block));
        }

        root["graph"] = std::move(graph);
    }

    root["version"] = std::string("2");
    root.erase("protocolVersion");
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

    Parser::Map meta;
    std::unordered_map<std::string, Parser::Map> blockMeta;

    Result resolveInputs(const TensorMap& requested, TensorMap& resolved) const;
    std::vector<std::string> collectDownstream(const std::string& name) const;
};

Result Flowgraph::Impl::resolveInputs(const TensorMap& requested, TensorMap& resolved) const {
    Result result = Result::SUCCESS;

    for (const auto& [slot, link] : requested) {
        if (!link.external.has_value()) {
            JST_ERROR("[FLOWGRAPH] Input '{}' has no external block reference.", slot);
            return Result::ERROR;
        }

        const auto& ext = link.external.value();

        if (!blocks.contains(ext.block)) {
            JST_ERROR("[FLOWGRAPH] Block '{}' does not exist.", ext.block);
            return Result::ERROR;
        }

        const auto& outputs = blocks.at(ext.block)->outputs();

        if (!outputs.contains(ext.port)) {
            JST_WARN("[FLOWGRAPH] Block '{}' has no output '{}' to satisfy connection '{}'.", ext.block,
                                                                                              ext.port,
                                                                                              slot);
            resolved.emplace(slot, link);
            result = Result::INCOMPLETE;
            continue;
        }

        resolved.emplace(slot, outputs.at(ext.port));
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
        JST_CHECK(impl->scheduler->stop());
        JST_CHECK(impl->scheduler->destroy());
    }

    impl->blocks.clear();
    impl->blockOrder.clear();
    impl->edges.clear();
    impl->path.clear();

    impl->created = false;
    return Result::SUCCESS;
}

Result Flowgraph::start() {
    if (!impl->created) {
        return Result::SUCCESS;
    }

    if (impl->scheduler) {
        JST_CHECK(impl->scheduler->start());
    }

    return Result::SUCCESS;
}

Result Flowgraph::stop() {
    if (!impl->created) {
        return Result::SUCCESS;
    }

    if (impl->scheduler) {
        JST_CHECK(impl->scheduler->stop());
    }

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
        if (!link.external.has_value()) {
            continue;
        }
        impl->edges[link.external->block].push_back(name);
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

    if (propagate && impl->edges.contains(name)) {
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
                if (!link.external.has_value()) {
                    continue;
                }
                if (link.external->block != name) {
                    state.inputs[slot].requested(link.external->block, link.external->port);
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
            state.inputs[slot].requested(link.external->block, link.external->port);
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
    newInputs[inputPort].requested(sourceBlock, sourcePort);

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
            state.inputs[slot].requested(link.external->block, link.external->port);
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
    if (!impl->blocks.contains(name)) {
        JST_ERROR("[FLOWGRAPH] Cannot recreate block '{}' because it doesn't exist.", name);
        return Result::ERROR;
    }

    const auto& block = impl->blocks.at(name);
    return blockRecreate(name, config, block->device(), block->runtime(), block->provider());
}

Result Flowgraph::blockRecreate(const std::string name,
                                const Parser::Map& config,
                                const DeviceType& device,
                                const RuntimeType& runtime,
                                const ProviderType& provider) {
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
        state.device = device;
        state.runtime = runtime;
        state.provider = provider;
        state.config = config;

        for (const auto& [slot, link] : block->inputs()) {
            state.inputs[slot].requested(link.external->block, link.external->port);
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
            state.inputs[slot].requested(link.external->block, link.external->port);
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
    Parser::Map root;
    JST_CHECK(Parser::YamlDecode(yamlText, root));

    JST_CHECK(MigrateFlowgraphVersion100To200(root));

    FlowgraphDocument document;
    JST_CHECK(document.deserialize(root));

    if (document.version != "2") {
        JST_ERROR("[FLOWGRAPH] Invalid flowgraph version '{}'.", document.version);
        return Result::ERROR;
    }

    if (document.title.has_value()) {
        impl->title = *document.title;
    }
    if (document.summary.has_value()) {
        impl->summary = *document.summary;
    }
    if (document.author.has_value()) {
        impl->author = *document.author;
    }
    if (document.license.has_value()) {
        impl->license = *document.license;
    }
    if (document.description.has_value()) {
        impl->description = *document.description;
    }

    if (document.meta.has_value()) {
        impl->meta = *document.meta;
    }

    if (document.graph.empty()) {
        return Result::SUCCESS;
    }

    struct NodeDef {
        std::string type;
        DeviceType device = DeviceType::CPU;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider = "generic";
        Parser::Map config;
        std::unordered_map<std::string, OutputRef> inputs;
        Parser::Map meta;
    };

    std::unordered_map<std::string, NodeDef> nodes;
    std::vector<std::string> nodeOrder;
    nodeOrder.reserve(document.graph.size());

    for (const auto& blockEntry : document.graph) {
        if (blockEntry.name.empty()) {
            JST_ERROR("[FLOWGRAPH] Block without a name found.");
            return Result::ERROR;
        }

        NodeDef def;
        def.type = blockEntry.module;
        def.device = blockEntry.device;
        def.runtime = blockEntry.runtime;
        def.provider = blockEntry.provider;
        if (blockEntry.config.has_value()) {
            def.config = *blockEntry.config;
        }
        if (blockEntry.meta.has_value()) {
            def.meta = *blockEntry.meta;
        }

        if (def.type.empty()) {
            JST_ERROR("[FLOWGRAPH] Block '{}' is missing 'module'.", blockEntry.name);
            return Result::ERROR;
        }
        if (blockEntry.input.has_value()) {
            for (const auto& [slot, encodedSource] : *blockEntry.input) {
                const auto& source = std::any_cast<const std::string&>(encodedSource);
                const auto ref = ParseGraphReference(source);
                if (!ref.has_value()) {
                    JST_ERROR("[FLOWGRAPH] Block '{}' input '{}' has invalid link '{}'.",
                              blockEntry.name,
                              slot,
                              source);
                    return Result::ERROR;
                }

                def.inputs[slot] = ref.value();
            }
        }

        if (nodes.contains(blockEntry.name)) {
            JST_ERROR("[FLOWGRAPH] Duplicate block '{}' found in flowgraph document.", blockEntry.name);
            return Result::ERROR;
        }

        nodes[blockEntry.name] = std::move(def);
        nodeOrder.push_back(blockEntry.name);
    }

    std::unordered_map<std::string, size_t> indegree;
    std::unordered_map<std::string, std::vector<std::string>> dependents;

    for (const auto& name : nodeOrder) {
        indegree[name] = 0;
    }

    for (const auto& name : nodeOrder) {
        const auto& def = nodes.at(name);
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
    for (const auto& name : nodeOrder) {
        if (indegree.at(name) == 0) {
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
            requestedInputs[slot].requested(ref.block, ref.port);
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

    FlowgraphDocument document;
    document.version = "2";
    if (!impl->title.empty()) {
        document.title = impl->title;
    }
    if (!impl->summary.empty()) {
        document.summary = impl->summary;
    }
    if (!impl->author.empty()) {
        document.author = impl->author;
    }
    if (!impl->license.empty()) {
        document.license = impl->license;
    }
    if (!impl->description.empty()) {
        document.description = impl->description;
    }
    document.meta = CompactMetaMap(impl->meta);

    const auto outputLookup = BuildOutputLookup(impl->blockOrder, impl->blocks);

    for (const auto& name : impl->blockOrder) {
        const auto& block = impl->blocks.at(name);
        FlowgraphBlockDocument blockDocument;

        blockDocument.name = name;
        blockDocument.module = block->config().type();
        blockDocument.device = block->device();
        blockDocument.runtime = block->runtime();
        blockDocument.provider = block->provider();

        Parser::Map config;
        JST_CHECK(block->config(config));
        if (!config.empty()) {
            blockDocument.config = std::move(config);
        }

        Parser::Map inputs;

        for (const auto& [slot, link] : block->inputs()) {
            const U64 tensorId = link.tensor.id();

            std::string producerBlock;
            std::string producerPort;

            if (link.external.has_value()) {
                producerBlock = link.external->block;
                producerPort = link.external->port;
            }

            if (tensorId != 0 && outputLookup.contains(tensorId)) {
                producerBlock = outputLookup.at(tensorId).block;
                producerPort = outputLookup.at(tensorId).port;
            }

            if (producerBlock.empty() || producerPort.empty()) {
                JST_ERROR("[FLOWGRAPH] Cannot resolve connection for input '{}.{}'.", name, slot);
                return Result::ERROR;
            }

            inputs[slot] = jst::fmt::format("${{graph.{}.output.{}}}",  producerBlock, producerPort);
        }

        if (!inputs.empty()) {
            blockDocument.input = std::move(inputs);
        }

        if (impl->blockMeta.contains(name)) {
            blockDocument.meta = CompactMetaMap(impl->blockMeta.at(name));
        }

        document.graph.push_back(std::move(blockDocument));
    }

    Parser::Map root;
    JST_CHECK(document.serialize(root));

    std::string yamlText;
    JST_CHECK(Parser::YamlEncode(root, yamlText));

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

Result Flowgraph::getMeta(const std::string& key, Parser::Map& data, const std::string& block) const {
    if (block.empty()) {
        if (impl->meta.contains(key) && impl->meta.at(key).type() == typeid(Parser::Map)) {
            data = std::any_cast<const Parser::Map&>(impl->meta.at(key));
        }
    } else if (impl->blockMeta.contains(block) &&
               impl->blockMeta.at(block).contains(key) &&
               impl->blockMeta.at(block).at(key).type() == typeid(Parser::Map)) {
        data = std::any_cast<const Parser::Map&>(impl->blockMeta.at(block).at(key));
    }
    return Result::SUCCESS;
}

Result Flowgraph::setMeta(const std::string& key, const Parser::Map& data, const std::string& block) {
    if (block.empty()) {
        impl->meta[key] = data;
    } else {
        impl->blockMeta[block][key] = data;
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream
