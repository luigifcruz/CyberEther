#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MESSAGES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MESSAGES_HH

#include "meta.hh"
#include "ui.hh"

#include "jetstream/instance_remote.hh"
#include "jetstream/module_surface.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/types.hh"

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace Jetstream {

struct MailNewFlowgraph {};

struct MailOpenFlowgraph {};

struct MailCloseFlowgraph {
    std::string flowgraph;
    bool force = false;
};

struct MailSaveFlowgraph {
    std::string flowgraph;
    std::string path;
};

struct MailOpenFlowgraphPath {
    std::string path;
};

struct MailOpenFlowgraphBlob {
    std::vector<char> blob;
};

struct MailFocusFlowgraph {
    std::string flowgraph;
};

struct MailApplyTheme {
    std::string themeKey;
};

struct MailOpenModal {
    ModalContent content;
    std::optional<SettingsSection> settings;
    std::optional<std::string> flowgraph;
};

struct MailSetSettingsSection {
    SettingsSection section = SettingsSection::General;
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

struct MailSetInfoPanelEnabled {
    bool value = false;
};

struct MailSetBackgroundParticles {
    bool value = false;
};

struct MailSetFlowgraphMetadataVisible {
    bool value = false;
};

struct MailSetFlowgraphMetadataSearch {
    std::string value;
};

struct MailSetFlowgraphEnvironmentVisible {
    bool value = false;
};

struct MailSetFlowgraphEnvironmentSearch {
    std::string value;
};

struct MailSetGraphicsScale {
    F32 value = 1.0f;
};

struct MailSetGraphicsDevice {
    DeviceType value = DeviceType::None;
};

struct MailSetGraphicsFramerate {
    U64 value = 60;
};

struct MailSetDebugLatencyEnabled {
    bool value = false;
};

struct MailSetDebugTimingEnabled {
    bool value = false;
};

struct MailSetDebugLogLevel {
    int value = JST_LOG_DEBUG_DEFAULT_LEVEL;
};

struct MailCheckForUpdates {};

struct MailDismissUpdate {};

struct MailSetRemoteBrokerUrl {
    std::string value;
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

struct MailSetRemoteAutoJoinSessions {
    bool value = false;
};

struct MailSetPythonRuntimePath {
    std::string value;
};

struct MailSaveFlowgraphPath {
    std::string flowgraph;
    std::string path;
};

struct MailBrowseConfigPath {
    std::string path;
    bool save = false;
    std::vector<std::string> extensions;
    std::function<void(std::string)> onSelect;
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
};

struct MailCreateStack {
    std::string flowgraph;
};

struct MailDeleteStack {
    std::string flowgraph;
    std::string stackId;
};

struct MailSetStackGeometry {
    std::string flowgraph;
    std::string stackId;
    F32 x = 0.0f;
    F32 y = 0.0f;
    F32 width = 0.0f;
    F32 height = 0.0f;
};

struct MailSetStackLayout {
    std::string flowgraph;
    std::string stackId;
    std::optional<StackDockLayoutMeta> layout;
};

struct MailSetSurfaceDetached {
    std::string flowgraph;
    std::string block;
    std::string surface;
    bool detached = false;
};

struct MailStartRemote {
    Instance::Remote::Config config;
};

struct MailStopRemote {};

struct MailApproveRemoteClient {
    std::string code;
};

struct MailSurfaceMouse {
    std::shared_ptr<Module::Surface> surface;
    MouseEvent event;
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
                          MailOpenFlowgraph,
                          MailCreateBlock,
                          MailCloseFlowgraph,
                          MailSaveFlowgraph,
                          MailOpenFlowgraphPath,
                          MailOpenFlowgraphBlob,
                          MailFocusFlowgraph,
                          MailApplyTheme,
                          MailOpenModal,
                          MailSetSettingsSection,
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
                          MailSetInfoPanelEnabled,
                          MailSetBackgroundParticles,
                          MailSetFlowgraphMetadataVisible,
                          MailSetFlowgraphMetadataSearch,
                          MailSetFlowgraphEnvironmentVisible,
                          MailSetFlowgraphEnvironmentSearch,
                          MailSetGraphicsScale,
                          MailSetGraphicsDevice,
                          MailSetGraphicsFramerate,
                          MailSetDebugLatencyEnabled,
                          MailSetDebugTimingEnabled,
                          MailSetDebugLogLevel,
                          MailCheckForUpdates,
                          MailDismissUpdate,
                          MailSetRemoteBrokerUrl,
                          MailSetRemoteCodec,
                          MailSetRemoteFramerate,
                          MailSetRemoteEncoder,
                          MailSetRemoteAutoJoinSessions,
                          MailSetPythonRuntimePath,
                          MailSaveFlowgraphPath,
                          MailBrowseConfigPath,
                          MailFilePickerNavigate,
                          MailFilePickerSelect,
                          MailFilePickerSetFilename,
                          MailFilePickerConfirm,
                          MailFilePickerCancel,
                          MailRunBenchmark,
                          MailResetBenchmark,
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
                          MailCreateStack,
                          MailDeleteStack,
                          MailSetStackGeometry,
                          MailSetStackLayout,
                          MailSetSurfaceDetached,
                          MailStartRemote,
                          MailStopRemote,
                          MailApproveRemoteClient,
                          MailSurfaceMouse,
                          MailResizeSurface>;

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MESSAGES_HH
