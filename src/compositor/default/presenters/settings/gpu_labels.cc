#include "gpu_labels.hh"

#include <algorithm>

namespace Jetstream {

std::string GpuLabels::Type(Backend::PhysicalDeviceType type) {
    switch (type) {
        case Backend::PhysicalDeviceType::DISCRETE: return "Discrete";
        case Backend::PhysicalDeviceType::INTEGRATED: return "Integrated";
        case Backend::PhysicalDeviceType::EXTERNAL: return "External";
        case Backend::PhysicalDeviceType::VIRTUAL: return "Virtual";
        case Backend::PhysicalDeviceType::CPU: return "CPU";
        case Backend::PhysicalDeviceType::OTHER: return "Other";
        case Backend::PhysicalDeviceType::UNKNOWN: break;
    }
    return "Unknown";
}

std::string GpuLabels::Selection(Backend::DeviceSelection selection) {
    switch (selection) {
        case Backend::DeviceSelection::Automatic: return "Automatic";
        case Backend::DeviceSelection::Saved: return "Saved";
        case Backend::DeviceSelection::Missing: return "Automatic (saved GPU not connected)";
        case Backend::DeviceSelection::Override: return "Device Index";
    }
    return "Automatic";
}

std::vector<std::string> GpuLabels::Names(const std::vector<Backend::PhysicalDevice>& devices) {
    std::vector<std::string> names;
    for (const auto& device : devices) {
        names.push_back(device.name);
    }

    for (U64 index = 0; index < devices.size(); ++index) {
        std::vector<U64> group;
        for (U64 other = 0; other < devices.size(); ++other) {
            if (devices[other].name == devices[index].name) {
                group.push_back(other);
            }
        }
        if (group.size() < 2) {
            continue;
        }
        std::stable_sort(group.begin(), group.end(), [&](U64 a, U64 b) {
            const auto& left = devices[a].pciAddress;
            const auto& right = devices[b].pciAddress;
            if (left.empty() != right.empty()) {
                return right.empty();
            }
            return left < right;
        });
        const auto position = std::find(group.begin(), group.end(), index) - group.begin();
        names[index] = jst::fmt::format("{} #{}", devices[index].name, position + 1);
    }

    return names;
}

GpuLabels::Choices GpuLabels::Build(const std::vector<Backend::PhysicalDevice>& devices, const Settings::Gpu& saved) {
    const auto names = Names(devices);
    const bool onlyCpu = std::all_of(devices.begin(), devices.end(), [](const auto& device) {
        return device.type == Backend::PhysicalDeviceType::CPU;
    });

    Choices choices;
    choices.options.push_back({
        .label = names.empty() ? std::string("Automatic") : jst::fmt::format("Automatic ({})", names.front()),
    });

    for (U64 index = 0; index < devices.size(); ++index) {
        const auto& device = devices[index];
        const bool hidden = device.type == Backend::PhysicalDeviceType::CPU && !onlyCpu;
        if (device.uuid.empty() || (hidden && device.uuid != saved.uuid)) {
            continue;
        }
        if (device.uuid == saved.uuid && choices.selected == 0) {
            choices.selected = choices.options.size();
        }
        choices.options.push_back({
            .label = names[index],
            .detail = Type(device.type),
            .value = {.uuid = device.uuid, .name = device.name},
        });
    }

    if (!saved.uuid.empty() && choices.selected == 0) {
        choices.selected = choices.options.size();
        choices.options.push_back({
            .label = jst::fmt::format("{} (not connected)", saved.name.empty() ? std::string("Saved GPU") : saved.name),
            .value = saved,
        });
    }

    choices.disabled = choices.options.size() == 1 ||
                       (choices.options.size() == 2 && saved.uuid.empty() && !devices.empty() &&
                        choices.options[1].value.uuid == devices.front().uuid);
    return choices;
}

}  // namespace Jetstream
