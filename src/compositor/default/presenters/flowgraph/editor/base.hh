#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_BASE_HH

#include "node.hh"
#include "picker.hh"

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../model/meta.hh"
#include "../../../views/flowgraph/editor/base.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"

#include <functional>
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

    explicit FlowgraphEditorPresenter(const PresenterContext& context) : context(context),
                                                                          node(context) {}

    FlowgraphEditor::Config build(const std::string& flowgraphId,
                                  const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
        FlowgraphEditor::Config config{
            .id = flowgraphId,
            .clipboardHasData = context.state.clipboard.hasData,
            .debugTimingEnabled = context.state.debug.timingEnabled,
            .blockOptions = catalog.buildBlockCatalog(),
            .title = flowgraph->title(),
            .summary = flowgraph->summary(),
            .author = flowgraph->author(),
            .license = flowgraph->license(),
            .description = flowgraph->description(),
            .path = flowgraph->path(),
            .onCreateBlock = [enqueue, flowgraphId](const std::string& moduleId,
                                                    std::optional<Extent2D<F32>> gridPosition,
                                                    DeviceType device,
                                                    RuntimeType runtime,
                                                    ProviderType provider) {
                enqueue(MailCreateBlock{flowgraphId, moduleId, gridPosition, device, runtime, provider});
            },
            .onCopyBlock = [enqueue, flowgraphId](const std::string& blockName) {
                enqueue(MailCopyBlock{flowgraphId, blockName});
            },
            .onPasteBlock = [enqueue, flowgraphId](std::optional<Extent2D<F32>> gridPosition) {
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
            .onChangeBlockDevice = [enqueue, flowgraphId](const std::string& blockName,
                                                          DeviceType device,
                                                          RuntimeType runtime,
                                                          ProviderType provider) {
                enqueue(MailChangeBlockDevice{flowgraphId, blockName, device, runtime, provider});
            },
            .onConnectBlock = [enqueue, flowgraphId](const std::string& blockName,
                                                     const std::string& inputPort,
                                                     const std::string& sourceBlock,
                                                     const std::string& sourcePort) {
                enqueue(MailConnectBlock{flowgraphId, blockName, inputPort, sourceBlock, sourcePort});
            },
            .onDisconnectBlock = [enqueue, flowgraphId](const std::string& blockName,
                                                        const std::string& inputPort) {
                enqueue(MailDisconnectBlock{flowgraphId, blockName, inputPort});
            },
            .onReconfigureBlock = [enqueue, flowgraphId](const std::string& blockName,
                                                         Parser::Map blockConfig,
                                                         const bool silent) {
                enqueue(MailReconfigureBlock{flowgraphId, blockName, std::move(blockConfig), silent});
            },
            .onConfigError = [enqueue](const Result result, const std::string& message) {
                enqueue(MailNotifyResult{.result = result, .message = message});
            },
            .onBrowseConfigPath = [enqueue](bool save,
                                            std::vector<std::string> extensions,
                                            std::function<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = "",
                    .save = save,
                    .extensions = std::move(extensions),
                    .onSelect = std::move(onSelect),
                });
            },
            .onNodeLayout = [enqueue, flowgraphId](const std::string& blockName,
                                                   F32 x,
                                                   F32 y,
                                                   F32 width,
                                                   F32 height) {
                enqueue(MailSetNodeMeta{flowgraphId, blockName, NodeMeta{x, y, width, height}});
            },
        };

        std::vector<std::string> blocks;
        if (flowgraph->view().keys(blocks) != Result::SUCCESS) {
            return config;
        }

        for (const auto& blockName : blocks) {
            Flowgraph::View::BlockData blockData;
            if (flowgraph->view().block(blockName, blockData) != Result::SUCCESS) {
                continue;
            }
            if (flowgraph->view().metrics(blockName, blockData.metrics) != Result::SUCCESS) {
                continue;
            }

            config.graph.push_back(node.build(flowgraphId, flowgraph, blockName, blockData));
        }

        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_BASE_HH
