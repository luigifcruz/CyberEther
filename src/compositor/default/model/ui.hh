#pragma once

#include <jetstream/types.hh>

namespace Jetstream {

enum class ModalContent : I32 {
    About = 0,
    FlowgraphInfo,
    FlowgraphClose,
    RenameBlock,
    Benchmark,
    Dependencies,
    Feedback,
    RemoteQr,
};

enum class RegistryPage : I32 {
    Overview = 0,
    AddPlugin,
    Plugin,
};

enum class IntelligencePage : I32 {
    Overview = 0,
    Provider,
    AddModel,
};

enum class WelcomeSection : I32 {
    All = 0,
    Flowgraphs,
};

enum class WorkspaceTab : I32 {
    Start = 0,
    Settings,
    Docs,
};

enum class SettingsSection : I32 {
    General = 0,
    Remote,
    Runtime,
    Registry,
    Intelligence,
    Developer,
    About,
    Legal,
};

}  // namespace Jetstream
