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
                                            const DeviceType& currentDevice) const {
        std::vector<DeviceType> devices;
        for (const auto& module : Registry::ListAvailableModules(blockType)) {
            if (std::find(devices.begin(), devices.end(), module.device) == devices.end()) {
                devices.push_back(module.device);
            }
        }

        if (currentDevice != DeviceType::None &&
            std::find(devices.begin(), devices.end(), currentDevice) == devices.end()) {
            devices.push_back(currentDevice);
        }

        return devices;
    }

    std::optional<Registry::ModuleRegistration> buildDeviceImplementation(const std::string& blockType,
                                                                          const DeviceType& device,
                                                                          const RuntimeType& preferredRuntime,
                                                                          const ProviderType& preferredProvider) const {
        const auto implementations = Registry::ListAvailableModules(blockType, std::optional<DeviceType>{device});
        if (implementations.empty()) {
            return std::nullopt;
        }

        const auto exactMatch = std::find_if(implementations.begin(), implementations.end(), [&](const auto& entry) {
            return entry.runtime == preferredRuntime && entry.provider == preferredProvider;
        });
        if (exactMatch != implementations.end()) {
            return *exactMatch;
        }

        const auto runtimeMatch = std::find_if(implementations.begin(), implementations.end(), [&](const auto& entry) {
            return entry.runtime == preferredRuntime;
        });
        if (runtimeMatch != implementations.end()) {
            return *runtimeMatch;
        }

        const auto providerMatch = std::find_if(implementations.begin(), implementations.end(), [&](const auto& entry) {
            return entry.provider == preferredProvider;
        });
        if (providerMatch != implementations.end()) {
            return *providerMatch;
        }

        return implementations.front();
    }

    std::vector<FlowgraphNode::DeviceOption> buildDeviceOptions(const std::string& blockType,
                                                                const DeviceType& currentDevice,
                                                                const RuntimeType& currentRuntime,
                                                                const ProviderType& currentProvider) const {
        std::vector<FlowgraphNode::DeviceOption> options;
        for (const auto& device : buildDeviceList(blockType, currentDevice)) {
            if (device == currentDevice) {
                options.push_back({
                    .label = GetDevicePrettyName(device),
                    .selected = true,
                    .device = currentDevice,
                    .runtime = currentRuntime,
                    .provider = currentProvider,
                });
                continue;
            }

            const auto implementation = buildDeviceImplementation(blockType,
                                                                  device,
                                                                  currentRuntime,
                                                                  currentProvider);
            if (!implementation.has_value()) {
                continue;
            }

            options.push_back({
                .label = GetDevicePrettyName(device),
                .selected = false,
                .device = implementation->device,
                .runtime = implementation->runtime,
                .provider = implementation->provider,
            });
        }

        return options;
    }

    std::vector<FlowgraphBlockPicker::BlockOption> buildBlockCatalog() const {
        std::vector<FlowgraphBlockPicker::BlockOption> options;
        for (const auto& entry : Registry::ListAvailableBlocks("")) {
            DeviceType device = DeviceType::CPU;
            RuntimeType runtime = RuntimeType::NATIVE;
            ProviderType provider = "generic";
            std::vector<FlowgraphBlockPicker::DeviceOption> devices;

            const auto modules = Registry::ListAvailableModules(entry.type);
            if (!modules.empty()) {
                device = modules.front().device;
                runtime = modules.front().runtime;
                provider = modules.front().provider;
                for (const auto& module : modules) {
                    const auto duplicate = std::find_if(devices.begin(), devices.end(), [&](const auto& option) {
                        return option.device == module.device;
                    });
                    if (duplicate == devices.end()) {
                        devices.push_back({
                            .device = module.device,
                            .runtime = module.runtime,
                            .provider = module.provider,
                        });
                    }
                }
            } else {
                devices.push_back({
                    .device = device,
                    .runtime = runtime,
                    .provider = provider,
                });
            }

            options.push_back({
                .type = entry.type,
                .title = entry.title,
                .summary = entry.summary,
                .description = entry.description,
                .category = entry.domain,
                .devices = std::move(devices),
                .device = device,
                .runtime = runtime,
                .provider = provider,
            });
        }

        return options;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_PICKER_HH
