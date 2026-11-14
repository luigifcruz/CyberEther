#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_ABOUT_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/about.hh"
#include "../../views/settings/components/about_info_table.hh"

#include "jetstream/backend/base.hh"
#include "jetstream/config.hh"
#include "jetstream/memory/types.hh"
#include "jetstream/types.hh"

#include <string>
#include <thread>
#include <vector>

namespace Jetstream {

struct AboutPresenter {
    struct Input {
        std::string viewportName;
        DeviceType device = DeviceType::None;
        Render::WindowMetrics metrics;
        Extent2D<F32> contentScale = {1.0f, 1.0f};
        Render::WindowMetrics::Policy scalePolicy = Render::WindowMetrics::Policy::Interface;
        bool integratedTitlebar = false;
        bool retainedPresent = false;
        std::string dependencyPolicy;
    };

    const PresenterContext& context;

    explicit AboutPresenter(const PresenterContext& context);
    AboutSettingsView::Config build() const;
    static std::string ThermalLabel(U64 state);
    static std::string DeviceTypeLabel(Backend::PhysicalDeviceType type);
    static std::string MemoryLabel(U64 bytes);
    static std::string PolicyLabel(const std::string& policy);
    static std::vector<AboutInfoTable::Config> InfoTables(const Input& input);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_ABOUT_HH
