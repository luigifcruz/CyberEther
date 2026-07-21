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
        for (const auto& target : Registry::ListAvailableBlockTargets(blockType)) {
            if (std::find(devices.begin(), devices.end(), target.device) ==
                devices.end()) {
                devices.push_back(target.device);
            }
        }

        if (currentDevice != DeviceType::None &&
            std::find(devices.begin(), devices.end(), currentDevice) == devices.end()) {
            devices.push_back(currentDevice);
        }

        return devices;
    }

    std::optional<Registry::BlockTarget> buildDeviceImplementation(const std::string& blockType,
                                                                   const DeviceType& device,
                                                                   const RuntimeType& preferredRuntime,
                                                                   const ProviderType& preferredProvider) const {
        auto implementations = Registry::ListAvailableBlockTargets(blockType);
        implementations.erase(
            std::remove_if(implementations.begin(), implementations.end(), [&](const auto& entry) {
                return entry.device != device;
            }),
            implementations.end());
        if (implementations.empty()) {
            return std::nullopt;
        }

        const auto exactMatch = std::find_if(
            implementations.begin(), implementations.end(), [&](const auto& entry) {
                return entry.runtime == preferredRuntime &&
                       entry.provider == preferredProvider;
            });
        if (exactMatch != implementations.end()) {
            return *exactMatch;
        }

        const auto runtimeMatch = std::find_if(
            implementations.begin(), implementations.end(), [&](const auto& entry) {
                return entry.runtime == preferredRuntime;
            });
        if (runtimeMatch != implementations.end()) {
            return *runtimeMatch;
        }

        const auto providerMatch = std::find_if(
            implementations.begin(), implementations.end(), [&](const auto& entry) {
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
            std::vector<Registry::BlockTarget> targets;

            for (const auto& target : Registry::ListAvailableBlockTargets(entry.type)) {
                const auto duplicate = std::find_if(
                    targets.begin(), targets.end(), [&](const auto& selected) {
                        return selected.device == target.device;
                    });
                if (duplicate == targets.end()) {
                    targets.push_back(target);
                }
            }
            if (targets.empty()) {
                continue;
            }

            std::vector<FlowgraphBlockPicker::DeviceOption> devices;
            devices.reserve(targets.size());
            for (const auto& target : targets) {
                devices.push_back({
                    .device = target.device,
                    .runtime = target.runtime,
                    .provider = target.provider,
                });
            }
            const auto defaultTarget = devices.front();

            options.push_back({
                .type = entry.type,
                .title = entry.title,
                .summary = entry.summary,
                .description = entry.description,
                .category = entry.domain,
                .devices = std::move(devices),
                .device = defaultTarget.device,
                .runtime = defaultTarget.runtime,
                .provider = defaultTarget.provider,
            });
        }

        return options;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_PICKER_HH
