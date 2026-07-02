#include <mutex>
#include <utility>

#include <jetstream/detail/flowgraph_impl.hh>

namespace Jetstream {

Flowgraph::View::View(const std::shared_ptr<Flowgraph::Impl>& impl) : impl(impl) {}

bool Flowgraph::View::has(const std::string& block) const {
    const auto graph = impl.lock();
    if (!graph) {
        return false;
    }

    std::lock_guard<std::recursive_mutex> lock(graph->blockMutex);
    return graph->blocks.contains(block) || graph->transientBlocks.contains(block);
}

bool Flowgraph::View::empty() const {
    const auto graph = impl.lock();
    if (!graph) {
        return true;
    }

    std::lock_guard<std::recursive_mutex> lock(graph->blockMutex);
    return graph->blocks.empty() && graph->transientBlocks.empty();
}

U64 Flowgraph::View::size() const {
    const auto graph = impl.lock();
    if (!graph) {
        return 0;
    }

    std::lock_guard<std::recursive_mutex> lock(graph->blockMutex);
    return graph->blocks.size() + graph->transientBlocks.size();
}

Result Flowgraph::View::keys(std::vector<std::string>& keys) const {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] View is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> lock(graph->blockMutex);

    keys.clear();
    keys.reserve(graph->blockOrder.size());
    for (const auto& name : graph->blockOrder) {
        if (graph->blocks.contains(name) || graph->transientBlocks.contains(name)) {
            keys.push_back(name);
        }
    }

    return Result::SUCCESS;
}

Result Flowgraph::View::info(const std::string& block, BlockInfo& info) const {
    BlockData data;
    JST_CHECK(this->block(block, data));

    info = data;
    return Result::SUCCESS;
}

Result Flowgraph::View::config(const std::string& block, Parser::Map& config) const {
    BlockData data;
    JST_CHECK(this->block(block, data));

    config = std::move(data.config);
    return Result::SUCCESS;
}

Result Flowgraph::View::inputs(const std::string& block, TensorMap& inputs) const {
    BlockData data;
    JST_CHECK(this->block(block, data));

    inputs = std::move(data.inputs);
    return Result::SUCCESS;
}

Result Flowgraph::View::outputs(const std::string& block, TensorMap& outputs) const {
    BlockData data;
    JST_CHECK(this->block(block, data));

    outputs = std::move(data.outputs);
    return Result::SUCCESS;
}

Result Flowgraph::View::interfaceInputs(const std::string& block,
                                        std::vector<InterfaceEntry>& inputs) const {
    BlockData data;
    JST_CHECK(this->block(block, data));

    inputs = std::move(data.interfaceInputs);
    return Result::SUCCESS;
}

Result Flowgraph::View::interfaceOutputs(const std::string& block,
                                         std::vector<InterfaceEntry>& outputs) const {
    BlockData data;
    JST_CHECK(this->block(block, data));

    outputs = std::move(data.interfaceOutputs);
    return Result::SUCCESS;
}

Result Flowgraph::View::interfaceConfigs(const std::string& block,
                                         std::vector<InterfaceEntry>& configs) const {
    BlockData data;
    JST_CHECK(this->block(block, data));

    configs = std::move(data.interfaceConfigs);
    return Result::SUCCESS;
}

Result Flowgraph::View::metrics(const std::string& block, std::vector<MetricEntry>& metrics) const {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] View is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> lock(graph->blockMutex);

    if (!graph->blocks.contains(block)) {
        if (graph->transientBlocks.contains(block)) {
            metrics.clear();
            return Result::SUCCESS;
        }

        JST_ERROR("[FLOWGRAPH] Block '{}' does not exist.", block);
        return Result::ERROR;
    }

    metrics.clear();

    const auto& interface = graph->blocks.at(block)->interface();
    if (!interface) {
        return Result::SUCCESS;
    }

    const auto& source = interface->metrics();
    metrics.reserve(source.size());
    for (const auto& [name, entry] : source) {
        std::any value;
        if (entry.metric) {
            value = entry.metric();
        }

        metrics.push_back({
            .name = name,
            .label = entry.label,
            .format = entry.format,
            .help = entry.help,
            .value = std::move(value),
        });
    }

    return Result::SUCCESS;
}

Result Flowgraph::View::surfaces(const std::string& block,
                                 std::vector<std::shared_ptr<Module::Surface>>& surfaces) const {
    BlockData data;
    JST_CHECK(this->block(block, data));

    surfaces = std::move(data.surfaces);
    return Result::SUCCESS;
}

Result Flowgraph::View::block(const std::string& block, BlockData& data) const {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] View is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> lock(graph->blockMutex);

    if (!graph->blocks.contains(block)) {
        if (graph->transientBlocks.contains(block)) {
            data = graph->transientBlocks.at(block);
            return Result::SUCCESS;
        }

        JST_ERROR("[FLOWGRAPH] Block '{}' does not exist.", block);
        return Result::ERROR;
    }

    const auto& blockPtr = graph->blocks.at(block);
    data = {};
    data.name = block;
    data.type = blockPtr->config().type();
    data.title = blockPtr->config().title();
    data.summary = blockPtr->config().summary();
    data.description = blockPtr->config().description();
    data.device = blockPtr->device();
    data.runtime = blockPtr->runtime();
    data.provider = blockPtr->provider();
    data.state = blockPtr->state();
    data.nodeSize = blockPtr->config().nodeSize();
    data.diagnostic = blockPtr->diagnostic();
    data.inputs = blockPtr->inputs();
    data.outputs = blockPtr->outputs();
    data.surfaces = blockPtr->surfaces();
    JST_CHECK(blockPtr->config(data.config));

    const auto& interface = blockPtr->interface();
    if (interface) {
        auto copyInterfaceEntries = [](const Block::Interface::EntryList& source,
                                       std::vector<InterfaceEntry>& target) {
            target.clear();
            target.reserve(source.size());
            for (const auto& [name, entry] : source) {
                target.push_back({
                    .name = name,
                    .label = entry.label,
                    .format = entry.format,
                    .help = entry.help,
                });
            }
        };

        copyInterfaceEntries(interface->inputs(), data.interfaceInputs);
        copyInterfaceEntries(interface->outputs(), data.interfaceOutputs);
        copyInterfaceEntries(interface->configs(), data.interfaceConfigs);
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
