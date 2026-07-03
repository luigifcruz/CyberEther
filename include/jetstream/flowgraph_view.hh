#ifndef JETSTREAM_FLOWGRAPH_VIEW_HH
#define JETSTREAM_FLOWGRAPH_VIEW_HH

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "jetstream/block_interface.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/module_surface.hh"
#include "jetstream/parser.hh"
#include "jetstream/tensor_link.hh"
#include "jetstream/types.hh"

namespace Jetstream {

class JETSTREAM_API Flowgraph::View {
 public:
    struct InterfaceEntry {
        std::string name;
        std::string label;
        std::string format;
        std::string help;
    };

    struct MetricEntry {
        std::string name;
        std::string label;
        std::string format;
        std::string help;
        std::any value;
    };

    struct BlockInfo {
        std::string name;
        std::string type;
        std::string title;
        std::string summary;
        std::string description;
        DeviceType device = DeviceType::CPU;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider = "generic";
        Block::State state = Block::State::None;
        Block::NodeSize nodeSize = Block::NodeSize::S;
        std::string diagnostic;
    };

    struct BlockData : BlockInfo {
        Parser::Map config;
        TensorMap inputs;
        TensorMap outputs;
        std::vector<InterfaceEntry> interfaceInputs;
        std::vector<InterfaceEntry> interfaceOutputs;
        std::vector<InterfaceEntry> interfaceConfigs;
        std::vector<MetricEntry> metrics;
        std::vector<std::shared_ptr<Module::Surface>> surfaces;
    };

    explicit View(const std::shared_ptr<Flowgraph::Impl>& impl);

    View(const View&) = delete;
    View& operator=(const View&) = delete;

    bool has(const std::string& block) const;
    bool empty() const;
    U64 size() const;

    Result keys(std::vector<std::string>& keys) const;
    Result info(const std::string& block, BlockInfo& info) const;
    Result config(const std::string& block, Parser::Map& config) const;
    Result inputs(const std::string& block, TensorMap& inputs) const;
    Result outputs(const std::string& block, TensorMap& outputs) const;
    Result interfaceInputs(const std::string& block, std::vector<InterfaceEntry>& inputs) const;
    Result interfaceOutputs(const std::string& block, std::vector<InterfaceEntry>& outputs) const;
    Result interfaceConfigs(const std::string& block, std::vector<InterfaceEntry>& configs) const;
    Result metrics(const std::string& block, std::vector<MetricEntry>& metrics) const;
    Result surfaces(const std::string& block,
                    std::vector<std::shared_ptr<Module::Surface>>& surfaces) const;
    Result block(const std::string& block, BlockData& data) const;

 private:
    std::weak_ptr<Flowgraph::Impl> impl;

    friend class Flowgraph;
};

}  // namespace Jetstream

#endif  // JETSTREAM_FLOWGRAPH_VIEW_HH
