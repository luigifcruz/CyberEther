#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_BASE_HH

#include "node.hh"
#include "picker.hh"

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../model/meta.hh"
#include "../../../support/terminal.hh"
#include "../../../views/flowgraph/editor/base.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphEditorPresenter {
    const PresenterContext& context;
    FlowgraphCatalogPresenter catalog;
    FlowgraphNodePresenter node;

    explicit FlowgraphEditorPresenter(const PresenterContext& context)
        : context(context), node(context) {}

    FlowgraphEditor::Config build(const std::string& flowgraphId,
                                  const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
        FlowgraphEditor::Config config{
            .id = flowgraphId,
            .clipboardHasData = context.state.clipboard.hasData,
            .debugTimingEnabled = context.state.debug.timingEnabled,
            .onBuildBlockOptions = [catalog = catalog]() {
                return catalog.buildBlockCatalog();
            },
            .onCreateBlock =
                [enqueue, flowgraphId](const std::string& moduleId,
                                       std::optional<Extent2D<F32>> gridPosition,
                                       DeviceType device,
                                       RuntimeType runtime,
                                       ProviderType provider) {
                    enqueue(MailCreateBlock{flowgraphId, moduleId, gridPosition,
                                            device, runtime, provider});
            },
            .onCopyBlock = [enqueue, flowgraphId](const std::string& blockName) {
                enqueue(MailCopyBlock{flowgraphId, blockName});
            },
            .onPasteBlock =
                [enqueue, flowgraphId](std::optional<Extent2D<F32>> gridPosition) {
                    enqueue(MailPasteBlock{flowgraphId, gridPosition});
            },
            .onRenameBlock = [enqueue, flowgraphId](const std::string& blockName) {
                enqueue(MailOpenRenameBlock{flowgraphId, blockName});
            },
            .onReloadBlock = [enqueue, flowgraphId](const std::string& blockName) {
                enqueue(MailReloadBlock{flowgraphId, blockName});
            },
            .onDeleteBlock = [enqueue, flowgraphId](const std::string& blockName) {
                enqueue(MailDeleteBlock{flowgraphId, blockName});
            },
            .onOpenDocumentation = [enqueue, flowgraphId](const std::string& blockName) {
                enqueue(MailOpenDocumentation{flowgraphId, blockName});
            },
            .onChangeBlockDevice = [enqueue, flowgraphId](const std::string& blockName,
                                                          DeviceType device,
                                                          RuntimeType runtime,
                                                          ProviderType provider) {
                enqueue(MailChangeBlockDevice{flowgraphId, blockName, device,
                                              runtime, provider});
            },
            .onConnectBlock = [enqueue, flowgraphId](const std::string& blockName,
                                                     const std::string& inputPort,
                                                     const std::string& sourceBlock,
                                                     const std::string& sourcePort) {
                enqueue(MailConnectBlock{flowgraphId, blockName, inputPort,
                                         sourceBlock, sourcePort});
            },
            .onDisconnectBlock = [enqueue, flowgraphId](const std::string& blockName,
                                                        const std::string& inputPort) {
                enqueue(MailDisconnectBlock{flowgraphId, blockName, inputPort});
            },
            .onReconfigureBlock = [enqueue, flowgraphId](const std::string& blockName,
                                                         Parser::Map blockConfig,
                                                         const bool silent,
                                                         Sakura::Callback<void(Result)> onComplete) {
                enqueue(MailReconfigureBlock{flowgraphId, blockName,
                                             std::move(blockConfig), silent,
                                             std::move(onComplete)});
            },
            .onConfigError =
                [enqueue](const Result result, const std::string& message) {
                    enqueue(MailNotifyResult{.result = result, .message = message});
            },
            .onBrowseConfigPath = [enqueue](bool save,
                                            std::vector<std::string> extensions,
                                            Sakura::Callback<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = "",
                    .save = save,
                    .extensions = std::move(extensions),
                    .onSelect = std::move(onSelect),
                });
            },
            .onOpenUrl = [enqueue](const std::string& url) {
                enqueue(MailOpenUrl{.url = url, .notifyResult = true});
            },
            .onNodeLayout = [enqueue, flowgraphId](const std::string& blockName,
                                                   F32 x,
                                                   F32 y,
                                                   F32 width,
                                                   F32 height,
                                                   std::optional<FlowgraphNode::Layout> base) {
                std::optional<NodeMeta> saved;
                if (base.has_value()) {
                    saved = NodeMeta{base->x, base->y, base->width, base->height};
                }
                enqueue(MailSetNodeMeta{
                    flowgraphId, blockName, NodeMeta{x, y, width, height}, saved});
            },
            .onNodeConfigCollapse = [enqueue, flowgraphId](const std::string& blockName,
                                                           bool collapsed) {
                enqueue(MailSetConfigCollapsed{
                    flowgraphId, blockName, collapsed});
            },
            .onNodeConfigDetach = [enqueue, flowgraphId](const std::string& blockName,
                                                         bool detached) {
                enqueue(MailSetConfigDetached{
                    flowgraphId, blockName, detached});
            },
        };

        const auto& agent = context.state.agent;
        config.measure = agent.chat.conversation.streaming || agent.voice.phase != Agent::Session::Phase::Idle;
        config.paused = flowgraph->paused();
        config.onTogglePause = [enqueue, flowgraphId, paused = config.paused]() {
            enqueue(MailSetFlowgraphPaused{.flowgraph = flowgraphId, .paused = !paused});
        };
        config.onSave = [enqueue, flowgraphId]() {
            enqueue(MailSaveFlowgraph{.flowgraph = flowgraphId});
        };
        config.onToggleEnvironment = [enqueue, flowgraphId,
                                      open = context.state.flowgraph.environment.contains(flowgraphId)]() {
            enqueue(MailSetFlowgraphEnvironmentVisible{.flowgraph = flowgraphId, .value = !open});
        };
        if (TerminalSession::Supported()) {
            config.onOpenTerminal = [enqueue, flowgraphId]() {
                enqueue(MailOpenTerminal{.flowgraph = flowgraphId, .tile = true});
            };
        }

        const auto blocks = context.state.flowgraph.blocks.find(flowgraphId);
        if (blocks == context.state.flowgraph.blocks.end()) {
            return config;
        }

        for (const auto& [blockName, blockData] : blocks->second) {
            config.graph.push_back(node.build(flowgraphId, flowgraph, blockName,
                                              blockData));
        }

        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_BASE_HH
