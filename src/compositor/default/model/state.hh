#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_STATE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_STATE_HH

#include "jetstream/benchmark.hh"
#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/instance.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/parser.hh"
#include "jetstream/plugin.hh"
#include "jetstream/registry.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/runtime_context_python.hh"
#include "jetstream/settings.hh"
#include "jetstream/viewport/adapters/generic.hh"

#include "render/sakura/runtime.hh"
#include "runtime/python/dependencies/coordinator.hh"

#include "ui.hh"
#include "meta.hh"
#include "file_picker.hh"
#include "surface_theme.hh"

#include "jetstream/agent/provider.hh"
#include "jetstream/agent/session.hh"
#include "jetstream/render/sakura/components/terminal.hh"
#include "jetstream/render/sakura/toast.hh"

#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct DefaultCompositorState {
    struct AgentState {
        struct ModelListResult {
            std::vector<Settings::Intelligence::Model> models;
            std::string error;
            std::map<std::string, Agent::Provider::Error::Kind> health;
        };

        struct ProviderKeyResult {
            bool success = false;
            std::string message;
        };

        struct ProviderKeyTask {
            enum class Kind {
                Save,
                Remove,
            };

            bool running = false;
            Kind kind = Kind::Save;
            U64 session = 0;
            Settings::Intelligence::Provider provider;
            std::string previousName;
            std::future<ProviderKeyResult> future;
        };

        struct ProvidersState {
            std::map<std::string, Agent::Provider::Error::Kind> health;
            std::optional<Settings::Intelligence::Provider> editTarget;
            U64 editSession = 0;
            ProviderKeyTask keyTask;
        };

        struct CatalogState {
            std::vector<Settings::Intelligence::Model> models;
            std::string search;
            bool loading = false;
            bool refreshPending = false;
            bool contextLengthsChecked = false;
            std::string error;
            std::future<ModelListResult> future;
            std::shared_ptr<std::atomic<bool>> cancel;
        };

        struct ProviderTestState {
            bool running = false;
            bool completed = false;
            bool success = false;
            U64 revision = 0;
            std::string message;
            std::future<std::pair<bool, std::string>> future;
            std::shared_ptr<std::atomic<bool>> cancel;
        };

        struct Conversation {
            U64 generation = 0;
            bool streaming = false;
            U64 contextTokens = 0;
            std::vector<std::shared_ptr<const Agent::Session::Entry>> messages;
        };

        struct ChatState {
            Conversation conversation;
            U64 dockAccepted = 0;
            U64 welcomeAccepted = 0;
        };

        struct VoiceState {
            Agent::Session::Phase phase = Agent::Session::Phase::Idle;
            F32 input = 0.0f;
            F32 output = 0.0f;
        };

        struct UiState {
            bool visible = false;
            bool unread = false;
            bool voiceVisible = false;
            bool settingsShown = false;
            U64 chatFocusRequests = 0;
        };

        struct McpState {
            std::string url;
            std::string error;
        };

        Settings::Intelligence settings;
        bool captureSupported = false;
        ProvidersState providers;
        CatalogState catalog;
        ProviderTestState providerTest;
        ChatState chat;
        VoiceState voice;
        UiState ui;
        McpState mcp;
    };

    struct SystemState {
        std::shared_ptr<Instance> instance;
        std::shared_ptr<Render::Window> render;
        std::shared_ptr<Viewport::Generic> viewport;
    };

    struct SakuraState {
        Sakura::Runtime runtime;
        std::string themeKey = "Dark";
        Sakura::Palette colorMap;

        bool dark() const {
            return themeKey != "Light";
        }
    };

    struct InterfaceState {
        static constexpr U64 MaxRecentFlowgraphs = 5;

        bool filePending = false;
        bool backgroundParticles = true;
        bool showTips = true;
        std::vector<Settings::Interface::FlowgraphEntry> recentFlowgraphs;
        std::vector<Settings::Interface::FlowgraphEntry> favoriteFlowgraphs;

        std::optional<std::string> focusedFlowgraph;
        std::optional<std::string> pendingFocusedFlowgraph;
        std::optional<std::string> closingFlowgraph;
        std::optional<WorkspaceTab> focusedTab;
        bool tabNavigationPending = false;
        bool startTab = false;
        U64 startTabBaseline = 0;
        U64 startTabRequests = 0;
        WelcomeSection welcomeSection = WelcomeSection::All;
        std::string welcomeQuery;
        std::string welcomeCategory;
        U64 welcomePage = 0;
        bool settingsTab = false;
        U64 settingsTabRequests = 0;
        bool docsTab = false;
        U64 docsTabRequests = 0;
        std::string docsPage;
        std::string docsAnchor;
        U64 docsNavigations = 0;
    };

    struct GraphicsState {
        std::optional<DeviceType> device;
        F32 scale = 1.0f;
    };

    struct SettingsState {
        SettingsSection section = SettingsSection::General;
        RegistryPage registryPage = RegistryPage::Overview;
        std::string registryPlugin;
        IntelligencePage intelligencePage = IntelligencePage::Overview;
    };

    struct RuntimeState {
        std::string pythonPath;
        std::string dependencyPolicy = "prompt";
        std::vector<PythonRuntimeContext::Candidate> pythonCandidates;
        PythonRuntimeContext::Validation pythonValidation;
        PythonRuntimeContext::Validation initialPythonValidation;
        PythonDependencyRequest dependencyRequest;
        U64 dependencyGeneration = 0;
    };

    struct ModalState {
        std::optional<ModalContent> content;
        std::optional<std::string> flowgraph;
        std::optional<std::string> renameBlockOldName;
        std::optional<PythonDependencyRequest> dependencyRequest;
    };

    struct DebugState {
        int logLevel = JST_LOG_DEBUG_DEFAULT_LEVEL;
        bool timingEnabled = false;
    };

    struct UpdateState {
        bool supported = false;
        bool upToDate = false;
        bool failed = false;
        bool checking = false;
        bool available = false;
        bool downloading = false;
        bool ready = false;
        bool applying = false;
        F32 progress = 0.0f;
        std::string version;
        std::string releaseNotes;
        std::string message;
    };

    struct FlowgraphState {
        struct BlockSnapshot {
            std::string name;
            Flowgraph::View::BlockData data;
        };

        struct Documentation {
            std::string block;
            std::string title;
            std::string markdown;
        };

        static std::string LiveMarkdownKey(const std::string& flowgraphId,
                                           const std::string& blockName,
                                           const std::string& fieldName) {
            return jst::fmt::format("{}:{}:{}", flowgraphId, blockName, fieldName);
        }

        std::unordered_map<std::string, std::shared_ptr<Flowgraph>> items;
        std::unordered_map<std::string, TileMeta> layouts;
        std::unordered_set<std::string> restoredLayouts;
        std::unordered_map<std::string, std::vector<BlockSnapshot>> blocks;
        std::unordered_map<std::string, std::string> liveMarkdown;
        std::unordered_map<std::string, Documentation> documentation;
        std::unordered_map<std::string, std::string> environment;
        std::unordered_map<std::string, std::string> gallery;
        SurfaceTracks surfaces;
    };

    struct TerminalState {
        struct Item {
            std::string title;
            std::string directory;
            std::optional<std::string> flowgraph;
            std::optional<U64> tab;
            Sakura::Terminal::Screen screen;
            U64 revision = 0;
            U64 focusRequests = 0;
            U64 focusedAt = 0;
            I32 zoom = 0;
            std::optional<std::string> search;
        };

        std::map<U64, Item> items;
        std::map<U64, TileMeta> tabs;
        U64 next = 1;
        U64 nextTab = 1;
        U64 focusSerial = 0;
        std::optional<U64> focused;
        std::optional<U64> requested;
        U64 requests = 0;
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
        std::string selectedModule;
        Benchmark::ResultMapType results;
        std::future<void> future;
        std::stringstream output;
    };

    struct RemoteState {
        Instance::Remote::Config config;

        bool supported = false;
        bool started = false;
        U64 clientCount = 0;
        std::string inviteUrl;
        std::vector<Instance::Remote::ClientInfo> clients;
        std::vector<std::string> waitlist;
    };

    struct RegistryState {
        struct BlockEntry {
            std::string type;
            std::string domain;
            std::string title;
            std::string summary;
        };

        struct PluginEntry {
            std::string path;
            std::optional<Plugin::Info> info;
            bool exists = false;
        };

        struct GalleryFetch {
            struct Fetched {
                std::vector<Registry::FlowgraphRegistration> flowgraphs;
                bool complete = true;
            };

            std::vector<Registry::FlowgraphRegistration> flowgraphs;
            std::future<std::optional<Fetched>> future;
            std::shared_ptr<std::atomic<bool>> cancel;
            std::chrono::steady_clock::time_point started;
            std::chrono::steady_clock::time_point next;
            bool cached = false;
            bool loading = false;
            bool attempted = false;
            bool failed = false;
        };

        bool stale = true;
        std::vector<BlockEntry> blocks;
        std::vector<Registry::FlowgraphRegistration> flowgraphs;
        GalleryFetch gallery;
        std::vector<PluginEntry> registered;
    };

    struct FilePickerState {
        bool active = false;
        bool overwritePending = false;
        U64 generation = 0;
        FilePickerMode mode = FilePickerMode::Open;
        std::string root;
        std::string directory;
        std::string selectedPath;
        std::string filename;
        std::string error;
        std::vector<std::string> extensions;
        std::vector<FilePickerEntry> entries;
    };

    struct FeedbackState {
        enum class Status {
            Idle,
            Submitting,
        };

        Status status = Status::Idle;
        std::string text;
    };

    SystemState system;
    SakuraState sakura;
    InterfaceState interface;
    GraphicsState graphics;
    SettingsState settings;
    AgentState agent;
    RuntimeState runtime;
    ModalState modal;
    DebugState debug;
    UpdateState update;
    FlowgraphState flowgraph;
    TerminalState terminal;
    BenchmarkState benchmark;
    ClipboardState clipboard;
    RemoteState remote;
    RegistryState registry;
    FilePickerState filePicker;
    FeedbackState feedback;

    bool startTabShown() const {
        return interface.startTab || workspaceTabs() == 0;
    }

    bool tabShown(WorkspaceTab tab) const {
        switch (tab) {
            case WorkspaceTab::Start:
                return startTabShown();
            case WorkspaceTab::Settings:
                return interface.settingsTab;
            case WorkspaceTab::Docs:
                return interface.docsTab;
        }
        return false;
    }

    bool tabActive(WorkspaceTab tab) const {
        return interface.focusedTab == tab && tabShown(tab);
    }

    bool startTabActive() const {
        return tabActive(WorkspaceTab::Start);
    }

    bool settingsTabActive() const {
        return tabActive(WorkspaceTab::Settings);
    }

    bool docsTabActive() const {
        return tabActive(WorkspaceTab::Docs);
    }

    std::optional<std::string> activeFlowgraph() const {
        if (startTabActive() || settingsTabActive() || docsTabActive() || terminal.focused.has_value()) {
            return std::nullopt;
        }
        if (!interface.focusedFlowgraph.has_value() || !flowgraph.items.contains(*interface.focusedFlowgraph)) {
            return std::nullopt;
        }
        return interface.focusedFlowgraph;
    }

    bool tabsHidden() const {
        return workspaceTabs() == 0 && !interface.startTab && !interface.settingsTab && !interface.docsTab;
    }

    U64 workspaceTabs() const {
        return flowgraph.items.size() + terminal.tabs.size();
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_STATE_HH
