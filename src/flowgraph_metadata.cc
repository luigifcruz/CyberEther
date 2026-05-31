#include <jetstream/detail/flowgraph_impl.hh>

#include <mutex>
#include <shared_mutex>

namespace Jetstream {

Flowgraph::Metadata::Metadata(const std::shared_ptr<Flowgraph::Impl>& impl) : impl(impl) {}

bool Flowgraph::Metadata::has(const std::string& key, const std::string& block) const {
    const auto graph = impl.lock();
    if (!graph) {
        return false;
    }

    std::shared_lock lock(graph->metadataMutex);

    if (block.empty()) {
        return graph->metadataValues.contains(key) &&
               graph->metadataValues.at(key).type() == typeid(Parser::Map);
    }

    return graph->blockMetadataValues.contains(block) &&
           graph->blockMetadataValues.at(block).contains(key) &&
           graph->blockMetadataValues.at(block).at(key).type() == typeid(Parser::Map);
}

Result Flowgraph::Metadata::get(const std::string& key, Parser::Map& data, const std::string& block) const {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] Metadata is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::shared_lock lock(graph->metadataMutex);

    if (block.empty()) {
        if (graph->metadataValues.contains(key) &&
            graph->metadataValues.at(key).type() == typeid(Parser::Map)) {
            data = std::any_cast<const Parser::Map&>(graph->metadataValues.at(key));
        }
    } else if (graph->blockMetadataValues.contains(block) &&
               graph->blockMetadataValues.at(block).contains(key) &&
               graph->blockMetadataValues.at(block).at(key).type() == typeid(Parser::Map)) {
        data = std::any_cast<const Parser::Map&>(graph->blockMetadataValues.at(block).at(key));
    }
    return Result::SUCCESS;
}

Result Flowgraph::Metadata::keys(std::vector<std::string>& keys, const std::string& block) const {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] Metadata is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    keys.clear();

    std::shared_lock lock(graph->metadataMutex);

    const Parser::Map* values = nullptr;
    if (block.empty()) {
        values = &graph->metadataValues;
    } else if (graph->blockMetadataValues.contains(block)) {
        values = &graph->blockMetadataValues.at(block);
    }

    if (values == nullptr) {
        return Result::SUCCESS;
    }

    keys.reserve(values->size());
    for (const auto& entry : *values) {
        if (entry.value.type() == typeid(Parser::Map)) {
            keys.push_back(entry.key);
        }
    }

    return Result::SUCCESS;
}

Result Flowgraph::Metadata::set(const std::string& key, const Parser::Map& data, const std::string& block) {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] Metadata is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::unique_lock lock(graph->metadataMutex);

    if (block.empty()) {
        graph->metadataValues[key] = data;
    } else {
        graph->blockMetadataValues[block][key] = data;
    }
    return Result::SUCCESS;
}

Result Flowgraph::Metadata::clear(const std::string& key, const std::string& block) {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] Metadata is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::unique_lock lock(graph->metadataMutex);

    if (block.empty()) {
        graph->metadataValues.erase(key);
    } else if (graph->blockMetadataValues.contains(block)) {
        graph->blockMetadataValues.at(block).erase(key);
        if (graph->blockMetadataValues.at(block).empty()) {
            graph->blockMetadataValues.erase(block);
        }
    }

    return Result::SUCCESS;
}

Result Flowgraph::Metadata::clearAll() {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] Metadata is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::unique_lock lock(graph->metadataMutex);

    graph->metadataValues.clear();
    graph->blockMetadataValues.clear();
    return Result::SUCCESS;
}

}  // namespace Jetstream
