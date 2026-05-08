#pragma once

#include <jetstream/types.hh>

#include <optional>

namespace Jetstream::Sakura {

struct DockspaceConfig {
    const char* id = "MainDockSpace";
    const char* windowTitle = "DockSpace";
    F32 topOffset = 0.0f;
    std::optional<U32> windowFlags;
    std::optional<U32> dockFlags;
};

U64 DockspaceId(const DockspaceConfig& config);
U64 DockspaceId();
void Dockspace(const DockspaceConfig& config);

}  // namespace Jetstream::Sakura
