#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_PICKER_HH

#include "../../../views/flowgraph/editor/node.hh"
#include "../../../views/flowgraph/editor/picker.hh"

#include "jetstream/registry.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

struct FlowgraphCatalogPresenter {
    std::vector<DeviceType> buildDeviceList(const std::string& blockType,
                                            const DeviceType& currentDevice) const;
    std::optional<Registry::BlockTarget> buildDeviceImplementation(const std::string& blockType,
                                                                   const DeviceType& device,
                                                                   const RuntimeType& preferredRuntime,
                                                                   const ProviderType& preferredProvider) const;
    std::vector<FlowgraphNode::DeviceOption> buildDeviceOptions(const std::string& blockType,
                                                                const DeviceType& currentDevice,
                                                                const RuntimeType& currentRuntime,
                                                                const ProviderType& currentProvider) const;
    std::vector<FlowgraphBlockPicker::BlockOption> buildBlockCatalog() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_PICKER_HH
