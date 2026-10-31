#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MESSAGES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MESSAGES_HH

#include "meta.hh"
#include "ui.hh"

#include "jetstream/instance_remote.hh"
#include "jetstream/module_surface.hh"
#include "jetstream/parser.hh"
#include "jetstream/settings.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/types.hh"

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace Jetstream {

struct MailSetAgentEnabled {
    bool value = true;
};

struct MailSetAgentChatVisible {
    bool value = false;
    bool focus = false;
};

struct MailSetAgentVoiceVisible {
    bool value = false;
};

struct MailStartAgentVoice {};

enum class AgentComposer : U8 {
    Dock,
    Welcome,
};

struct MailSubmitAgentChat {
    AgentComposer composer = AgentComposer::Dock;
    std::string text;
    bool reveal = false;
};

struct MailCancelAgentChat {};
struct MailClearAgentChat {};

struct MailSelectAgentModel {
    std::string provider;
    std::string model;
};

struct MailSetAgentReasoningEffort {
    std::string provider;
    std::string model;
    std::string value;
};

struct MailOpenAgentModelPicker {};

struct MailSetAgentModelSearch {
    std::string value;
};

struct MailAddAgentModel {
    Settings::Intelligence::Model value;
};

struct MailRemoveAgentModel {
    Settings::Intelligence::Model value;
};

struct MailOpenAgentProviderEditor {
    std::optional<std::string> name;
};

struct MailSaveAgentProvider {
    Settings::Intelligence::Provider value;
    std::optional<std::string> replaceName;
    std::string apiKey;
};

struct MailRemoveAgentProvider {
    std::string name;
};

struct MailTestAgentProvider {
    Settings::Intelligence::Provider value;
    U64 revision = 0;
    std::string apiKey;
};

struct MailSetAgentVoiceBackend {
    Settings::Intelligence::Model value;
};

struct MailSetAgentVoiceModel {
    std::string value;
};

struct MailSetAgentVoiceName {
    std::string value;
};

struct MailSetAgentMcp {
    Settings::Intelligence::Mcp value;
};

struct MailNewFlowgraph {};

struct MailOpenStartTab {};

struct MailCloseStartTab {};

struct MailFocusStartTab {};

struct MailSetWelcomeSection {
    WelcomeSection section = WelcomeSection::All;
};

struct MailSetWelcomeSidebar {
    bool open = true;
};

struct MailSetWelcomeQuery {
    std::string query;
};

struct MailSetWelcomeCategory {
    std::string category;
};

struct MailSetWelcomePage {
    U64 page = 0;
};

struct MailOpenSettings {
    SettingsSection section = SettingsSection::General;
};

struct MailCloseSettings {};

struct MailFocusSettings {};

struct MailOpenDocs {
    std::string page;
    std::string anchor;
};

struct MailCloseDocs {};

struct MailSetDocsQuery {
    std::string query;
};

struct MailFocusDocs {};

struct MailOpenFlowgraph {};

struct MailCloseFlowgraph {
    std::string flowgraph;
    bool force = false;
};

struct MailCancelCloseFlowgraph {};

struct MailSaveFlowgraph {
    std::string flowgraph;
    std::string path;
};

struct MailOpenFlowgraphPath {
    std::string path;
};

struct MailToggleFavoriteFlowgraph {
    std::string path;
};

struct MailRememberFlowgraph {
    std::string path;
    std::string title;
};

struct MailDropFlowgraphPath {
    std::string path;
};

struct MailOpenFlowgraphBlob {
    std::vector<char> blob;
};

struct MailOpenFlowgraphExample {
    std::string key;
};

struct MailFocusFlowgraph {
    std::string flowgraph;
};

struct MailSetFlowgraphPaused {
    std::string flowgraph;
    bool paused = false;
};

struct MailApplyTheme {
    std::string themeKey;
};

struct MailOpenModal {
    ModalContent content;
    std::optional<std::string> flowgraph;
};

struct MailSetSettingsSection {
    SettingsSection section = SettingsSection::General;
};

struct MailSetIntelligencePage {
    IntelligencePage page = IntelligencePage::Overview;
};

struct MailSetRegistryPage {
    RegistryPage page = RegistryPage::Overview;
    std::string plugin;
};

struct MailAddPluginPath {
    std::string path;
};

struct MailRemovePluginPath {
    std::string path;
};

struct MailReloadPlugin {
    std::string path;
};

struct MailReloadAllPlugins {};

struct MailCloseModal {};

struct MailNotify {
    Sakura::ToastType type;
    I32 durationMs = 0;
    std::string message;
};

struct MailNotifyResult {
    Result result;
    std::string message;
};

struct MailOpenUrl {
    std::string url;
    bool notifyResult = false;
};

struct MailCopyText {
    std::string label;
    std::string value;
};

struct MailQuit {};

struct MailSetBackgroundParticles {
    bool value = false;
};

struct MailSetShowTips {
    bool value = false;
};

struct MailSetFlowgraphEnvironmentVisible {
    std::string flowgraph;
    bool value = false;
};

struct MailSetFlowgraphEnvironmentSearch {
    std::string flowgraph;
    std::string value;
};

struct MailSetGraphicsScale {
    F32 value = 1.0f;
};

struct MailSetGraphicsDevice {
    DeviceType value = DeviceType::None;
};

struct MailSetDebugTimingEnabled {
    bool value = false;
};

struct MailSetDebugLogLevel {
    int value = JST_LOG_DEBUG_DEFAULT_LEVEL;
};

struct MailCheckForUpdates {};

struct MailDownloadUpdate {};

struct MailApplyUpdate {};

struct MailSetRemoteBroker {
    Instance::Remote::Config::Broker value;
};

struct MailSetRemoteCodec {
    Instance::Remote::CodecType value = Instance::Remote::CodecType::H264;
};

struct MailSetRemoteFramerate {
    U32 value = 30;
};

struct MailSetRemoteEncoder {
    Instance::Remote::EncoderType value = Instance::Remote::EncoderType::Auto;
};

struct MailSetRemoteAutoApprove {
    bool value = false;
};

struct MailSetRemoteTransport {
    Instance::Remote::TransportType value = Instance::Remote::TransportType::Auto;
};

struct MailSetRemotePortRange {
    U16 minPort = 0;
    U16 maxPort = 0;
};

struct MailSetRemoteMediaAddress {
    std::string value;
};

struct MailSetPythonRuntimePath {
    std::string value;
};

struct MailSetRuntimeDependencyPolicy {
    std::string value;
};

struct MailOpenPythonEnvironmentCache {};

struct MailInstallPythonDependencies {
    U64 generation = 0;
};

struct MailSaveFlowgraphPath {
    std::string flowgraph;
    std::string path;
};

struct MailBrowseConfigPath {
    std::string path;
    bool save = false;
    std::vector<std::string> extensions;
    Sakura::Callback<void(std::string)> onSelect;
};

struct MailFilePickerNavigate {
    U64 generation = 0;
    std::string path;
};

struct MailFilePickerSelect {
    U64 generation = 0;
    std::string path;
};

struct MailFilePickerSetFilename {
    U64 generation = 0;
    std::string value;
};

struct MailFilePickerConfirm {
    U64 generation = 0;
};

struct MailFilePickerCancel {
    U64 generation = 0;
};

struct MailRunBenchmark {};

struct MailResetBenchmark {};

struct MailSetBenchmarkModule {
    std::string moduleType;
};

struct MailSetFeedbackText {
    std::string value;
};

struct MailSubmitFeedback {};

struct MailSetFlowgraphInfo {
    std::string flowgraph;
    std::optional<std::string> title;
    std::optional<std::string> summary;
    std::optional<std::string> author;
    std::optional<std::string> license;
    std::optional<std::string> description;
};

struct MailCreateBlock {
    std::string flowgraph;
    std::string moduleId;
    std::optional<Extent2D<F32>> gridPosition;
    DeviceType device;
    RuntimeType runtime;
    ProviderType provider;
};

struct MailRenameBlock {
    std::string flowgraph;
    std::string oldId;
    std::string newId;
};

struct MailOpenRenameBlock {
    std::string flowgraph;
    std::string blockId;
};

struct MailDeleteBlock {
    std::string flowgraph;
    std::string blockId;
};

struct MailReloadBlock {
    std::string flowgraph;
    std::string blockId;
};

struct MailChangeBlockDevice {
    std::string flowgraph;
    std::string blockId;
    DeviceType device;
    RuntimeType runtime;
    ProviderType provider;
};

struct MailConnectBlock {
    std::string flowgraph;
    std::string blockName;
    std::string inputPort;
    std::string sourceBlock;
    std::string sourcePort;
};

struct MailDisconnectBlock {
    std::string flowgraph;
    std::string blockName;
    std::string inputPort;
};

struct MailReconfigureBlock {
    std::string flowgraph;
    std::string blockId;
    Parser::Map config;
    bool silent = false;
    Sakura::Callback<void(Result)> onComplete;
};

struct MailCopyBlock {
    std::string flowgraph;
    std::string blockId;
};

struct MailPasteBlock {
    std::string flowgraph;
    std::optional<Extent2D<F32>> gridPosition;
};

struct MailSetNodeMeta {
    std::string flowgraph;
    std::string block;
    NodeMeta meta;
    std::optional<NodeMeta> base;
};

struct MailSetConfigCollapsed {
    std::string flowgraph;
    std::string block;
    bool collapsed = false;
};

struct MailSetConfigDetached {
    std::string flowgraph;
    std::string block;
    bool detached = false;
};

struct MailSetFlowgraphLayout {
    std::string flowgraph;
    std::optional<TileMeta> tiles;
};

struct MailRemoveLayoutBlock {
    std::string flowgraph;
    std::string block;
};

struct MailRenameLayoutBlock {
    std::string flowgraph;
    std::string oldId;
    std::string newId;
};

struct MailOpenDocumentation {
    std::string flowgraph;
    std::string block;
};

struct MailCloseDocumentation {
    std::string flowgraph;
};

struct MailCloseFlowgraphDescription {
    std::string flowgraph;
};

struct MailSetSurfaceDetached {
    std::string flowgraph;
    std::string block;
    std::string surface;
    bool detached = false;
};

struct MailSetSurfaceConfigCollapsed {
    std::string flowgraph;
    std::string block;
    std::string surface;
    bool collapsed = true;
};

struct MailStartRemote {
    Instance::Remote::Config config;
};

struct MailStopRemote {};

struct MailApproveRemoteClient {
    std::string sessionId;
};

struct MailRemoveRemoteClient {
    std::string sessionId;
};

struct MailSurfaceInput {
    std::shared_ptr<Module::Surface> surface;
    InputEvent event;
};

enum class SurfacePlacement {
    Attached,
    Detached,
};

struct SurfaceResizePayload {
    Extent2D<U64> logicalSize = {0, 0};
    Extent2D<U64> framebufferSize = {0, 0};
    F32 scale = 1.0f;
};

struct MailResizeSurface {
    std::shared_ptr<Module::Surface> surface;
    std::string flowgraph;
    std::string block;
    std::string metaKey;
    SurfacePlacement placement = SurfacePlacement::Attached;
    SurfaceResizePayload resize;
};

using Mail = std::variant<MailNewFlowgraph,
                          MailOpenStartTab,
                          MailCloseStartTab,
                          MailFocusStartTab,
                          MailSetWelcomeSection,
                          MailSetWelcomeSidebar,
                          MailSetWelcomeQuery,
                          MailSetWelcomeCategory,
                          MailSetWelcomePage,
                          MailOpenSettings,
                          MailCloseSettings,
                          MailFocusSettings,
                          MailOpenDocs,
                          MailCloseDocs,
                          MailSetDocsQuery,
                          MailFocusDocs,
                          MailOpenFlowgraph,
                          MailCreateBlock,
                          MailCloseFlowgraph,
                          MailCancelCloseFlowgraph,
                          MailSaveFlowgraph,
                          MailOpenFlowgraphPath,
                          MailToggleFavoriteFlowgraph,
                          MailRememberFlowgraph,
                          MailDropFlowgraphPath,
                          MailOpenFlowgraphBlob,
                          MailOpenFlowgraphExample,
                          MailFocusFlowgraph,
                          MailSetFlowgraphPaused,
                          MailApplyTheme,
                          MailOpenModal,
                          MailSetSettingsSection,
                          MailSetRegistryPage,
                          MailSetIntelligencePage,
                          MailAddPluginPath,
                          MailRemovePluginPath,
                          MailReloadPlugin,
                          MailReloadAllPlugins,
                          MailCloseModal,
                          MailNotify,
                          MailNotifyResult,
                          MailOpenUrl,
                          MailCopyText,
                          MailQuit,
                          MailSetBackgroundParticles,
                          MailSetShowTips,
                          MailSetFlowgraphEnvironmentVisible,
                          MailSetFlowgraphEnvironmentSearch,
                          MailSetGraphicsScale,
                          MailSetGraphicsDevice,
                          MailSetAgentEnabled,
                          MailSetAgentChatVisible,
                          MailSetAgentVoiceVisible,
                          MailStartAgentVoice,
                          MailSubmitAgentChat,
                          MailCancelAgentChat,
                          MailClearAgentChat,
                          MailSelectAgentModel,
                          MailSetAgentReasoningEffort,
                          MailOpenAgentModelPicker,
                          MailSetAgentModelSearch,
                          MailAddAgentModel,
                          MailRemoveAgentModel,
                          MailOpenAgentProviderEditor,
                          MailSaveAgentProvider,
                          MailRemoveAgentProvider,
                          MailTestAgentProvider,
                          MailSetAgentVoiceBackend,
                          MailSetAgentVoiceModel,
                          MailSetAgentVoiceName,
                          MailSetAgentMcp,
                          MailSetDebugTimingEnabled,
                          MailSetDebugLogLevel,
                          MailCheckForUpdates,
                          MailDownloadUpdate,
                          MailApplyUpdate,
                          MailSetRemoteBroker,
                          MailSetRemoteCodec,
                          MailSetRemoteFramerate,
                          MailSetRemoteEncoder,
                          MailSetRemoteAutoApprove,
                          MailSetRemoteTransport,
                          MailSetRemotePortRange,
                          MailSetRemoteMediaAddress,
                          MailSetPythonRuntimePath,
                          MailSetRuntimeDependencyPolicy,
                          MailOpenPythonEnvironmentCache,
                          MailInstallPythonDependencies,
                          MailSaveFlowgraphPath,
                          MailBrowseConfigPath,
                          MailFilePickerNavigate,
                          MailFilePickerSelect,
                          MailFilePickerSetFilename,
                          MailFilePickerConfirm,
                          MailFilePickerCancel,
                          MailRunBenchmark,
                          MailResetBenchmark,
                          MailSetBenchmarkModule,
                          MailSetFeedbackText,
                          MailSubmitFeedback,
                          MailSetFlowgraphInfo,
                          MailOpenRenameBlock,
                          MailRenameBlock,
                          MailDeleteBlock,
                          MailReloadBlock,
                          MailChangeBlockDevice,
                          MailConnectBlock,
                          MailDisconnectBlock,
                          MailReconfigureBlock,
                          MailCopyBlock,
                          MailPasteBlock,
                          MailSetNodeMeta,
                          MailSetConfigCollapsed,
                          MailSetConfigDetached,
                          MailSetFlowgraphLayout,
                          MailRemoveLayoutBlock,
                          MailRenameLayoutBlock,
                          MailOpenDocumentation,
                          MailCloseDocumentation,
                          MailCloseFlowgraphDescription,
                          MailSetSurfaceDetached,
                          MailSetSurfaceConfigCollapsed,
                          MailStartRemote,
                          MailStopRemote,
                          MailApproveRemoteClient,
                          MailRemoveRemoteClient,
                          MailSurfaceInput,
                          MailResizeSurface>;

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MESSAGES_HH
