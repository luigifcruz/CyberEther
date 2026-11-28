#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_GPU_LABELS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_GPU_LABELS_HH

#include "jetstream/backend/config.hh"
#include "jetstream/settings.hh"
#include "jetstream/types.hh"

#include <string>
#include <vector>

namespace Jetstream {

class GpuLabels {
 public:
    struct Option {
        std::string label;
        std::string detail;
        Settings::Gpu value;

        bool operator==(const Option&) const = default;
    };

    struct Choices {
        std::vector<Option> options;
        U64 selected = 0;
        bool disabled = true;

        bool operator==(const Choices&) const = default;
    };

    static std::string Type(Backend::PhysicalDeviceType type);
    static std::string Selection(Backend::DeviceSelection selection);
    static std::vector<std::string> Names(const std::vector<Backend::PhysicalDevice>& devices);
    static Choices Build(const std::vector<Backend::PhysicalDevice>& devices, const Settings::Gpu& saved);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_GPU_LABELS_HH
