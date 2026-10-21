#pragma once

#include <jetstream/types.hh>

namespace Jetstream {

enum class ModalContent : I32 {
    About = 0,
    FlowgraphExamples,
    FlowgraphInfo,
    FlowgraphClose,
    RenameBlock,
    Benchmark,
    RemoteStreaming,
    Dependencies,
    Feedback,
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
