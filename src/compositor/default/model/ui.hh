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
    Settings,
};

enum class SettingsSection : I32 {
    General = 0,
    Remote,
    Developer,
    About,
    Legal,
};

}  // namespace Jetstream
