#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_STATE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_STATE_HH

#include "jetstream/benchmark.hh"
#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/instance.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/runtime_context_python.hh"
#include "jetstream/viewport/adapters/generic.hh"

#include "render/sakura/runtime.hh"

#include "ui.hh"
#include "meta.hh"

#include <future>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace Jetstream {

struct DefaultCompositorState {
    struct SystemState {
        std::shared_ptr<Instance> instance;
        std::shared_ptr<Render::Window> render;
        std::shared_ptr<Viewport::Generic> viewport;
    };

    struct SakuraState {
        Sakura::Runtime runtime;
        std::string themeKey = "Dark";
        Sakura::Palette colorMap;
    };

    struct InterfaceState {
        bool filePending = false;
        bool infoPanelEnabled = true;
        bool backgroundParticles = true;
        bool flowgraphMetadataVisible = false;
        std::string flowgraphMetadataSearch;
        bool flowgraphEnvironmentVisible = false;
        std::string flowgraphEnvironmentSearch;

        std::optional<std::string> focusedFlowgraph;
        std::optional<std::string> pendingFocusedFlowgraph;
    };

    struct GraphicsState {
        std::optional<DeviceType> device;
        F32 scale = 1.0f;
        U64 framerate = 60;
    };

    struct SettingsState {
        SettingsSection section = SettingsSection::General;
    };

    struct RuntimeState {
        std::string pythonPath;
        std::vector<PythonRuntimeContext::Candidate> pythonCandidates;
        PythonRuntimeContext::Validation pythonValidation;
        PythonRuntimeContext::Validation initialPythonValidation;
    };

    struct ModalState {
        std::optional<ModalContent> content;
        std::optional<std::string> flowgraph;
        std::optional<std::string> renameBlockOldName;
    };

    struct DebugState {
        int logLevel = JST_LOG_DEBUG_DEFAULT_LEVEL;
        bool latencyEnabled = false;
        bool timingEnabled = false;
    };

    struct UpdateState {
        bool checking = false;
        bool available = false;
        std::string version;
    };

    struct FlowgraphState {
        struct StackWindowState {
            StackMeta meta;
            bool restoreDockLayout = false;
            bool dockInMainDockspace = true;
        };

        std::unordered_map<std::string, std::shared_ptr<Flowgraph>> items;
        std::unordered_map<std::string, std::unordered_map<std::string, StackWindowState>> stacks;
    };

    struct ClipboardState {
        std::string moduleType;
        DeviceType device;
        RuntimeType runtime;
        ProviderType provider;
        Parser::Map config;
        bool hasData = false;
    };

    struct BenchmarkState {
        bool running = false;
        F32 progress = 0.0f;
        Benchmark::ResultMapType results;
        std::future<void> future;
        std::stringstream output;
    };

    struct RemoteState {
        std::string brokerUrl = "https://cyberether.org";
        Instance::Remote::CodecType codec = Instance::Remote::CodecType::H264;
        Instance::Remote::EncoderType encoder = Instance::Remote::EncoderType::Auto;
        bool autoJoinSessions = false;
        U32 framerate = 30;

        bool supported = false;
        bool started = false;
        U64 clientCount = 0;
        std::string inviteUrl;
        std::string roomId;
        std::string accessToken;
        std::vector<Instance::Remote::ClientInfo> clients;
        std::vector<std::string> waitlist;
    };

    SystemState system;
    SakuraState sakura;
    InterfaceState interface;
    GraphicsState graphics;
    SettingsState settings;
    RuntimeState runtime;
    ModalState modal;
    DebugState debug;
    UpdateState update;
    FlowgraphState flowgraph;
    BenchmarkState benchmark;
    ClipboardState clipboard;
    RemoteState remote;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_STATE_HH
