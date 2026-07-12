#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/instance.hh"
#include "jetstream/platform.hh"

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace Jetstream {

struct FlowgraphActions {
    using Filter = std::tuple<MailNewFlowgraph,
                              MailOpenFlowgraph,
                              MailOpenFlowgraphPath,
                              MailOpenFlowgraphBlob,
                              MailFocusFlowgraph,
                              MailSaveFlowgraph,
                              MailCloseFlowgraph,
                              MailSaveFlowgraphPath,
                              MailSetFlowgraphInfo,
                              MailCreateBlock,
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
                              MailSurfaceMouse,
                              MailResizeSurface>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    FlowgraphActions(DefaultCompositorState& state,
                     DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    Result handle(const MailNewFlowgraph&) {
        auto inst = state.system.instance;
        auto name = GenerateRandomFlowgraphName();

        callbacks.enqueueCommand([inst, name]() -> Result {
            std::shared_ptr<Flowgraph> flowgraph;
            JST_CHECK(inst->flowgraphCreate(name, {}, flowgraph));
            return Result::SUCCESS;
        }, false);

        state.interface.pendingFocusedFlowgraph = name;
        return Result::SUCCESS;
    }

    Result handle(const MailOpenFlowgraph&) {
        std::string path;
        auto enqueueMail = callbacks.enqueueMail;
        callbacks.requestFile({
            .mode = FilePickerMode::Open,
            .initialPath = path,
            .extensions = {"yaml", "yml"},
            .callback = [enqueueMail](std::string p) mutable {
                enqueueMail(MailOpenFlowgraphPath{std::move(p)});
            },
        });

        return Result::SUCCESS;
    }

    Result handle(const MailOpenFlowgraphPath& msg) {
        if (msg.path.empty()) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Cannot open flowgraph due to empty path.");
            return Result::SUCCESS;
        }

        if (!std::filesystem::exists(Platform::PathFromUtf8(msg.path))) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "The selected file does not exist.");
            return Result::SUCCESS;
        }

        auto inst = state.system.instance;
        auto name = GenerateRandomFlowgraphName();
        auto path = msg.path;

        callbacks.enqueueCommand([inst, name, path]() -> Result {
            std::shared_ptr<Flowgraph> flowgraph;
            JST_CHECK(inst->flowgraphCreate(name, {}, flowgraph));
            JST_CHECK(flowgraph->importFromFile(path));
            return Result::SUCCESS;
        }, false);

        state.interface.pendingFocusedFlowgraph = name;
        return Result::SUCCESS;
    }

    Result handle(const MailOpenFlowgraphBlob& msg) {
        if (msg.blob.empty()) {
            JST_ERROR("Failed to open flowgraph from blob due to invalid blob data.");
            return Result::ERROR;
        }

        auto inst = state.system.instance;
        auto name = GenerateRandomFlowgraphName();
        auto blob = msg.blob;

        callbacks.enqueueCommand([inst, name, blob]() -> Result {
            std::shared_ptr<Flowgraph> flowgraph;
            JST_CHECK(inst->flowgraphCreate(name, {}, flowgraph));
            JST_CHECK(flowgraph->importFromBlob(blob));
            return Result::SUCCESS;
        }, false);

        state.interface.pendingFocusedFlowgraph = name;
        return Result::SUCCESS;
    }

    Result handle(const MailFocusFlowgraph& msg) {
        if (state.flowgraph.items.contains(msg.flowgraph)) {
            state.interface.focusedFlowgraph = msg.flowgraph;
        }

        return Result::SUCCESS;
    }

    Result handle(const MailSaveFlowgraph& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Cannot save flowgraph because it doesn't exist.");
            return Result::SUCCESS;
        }

        std::string path = msg.path.empty() ? state.flowgraph.items.at(msg.flowgraph)->path() : msg.path;
        if (path.empty()) {
            std::string pickedPath;
            auto enqueueMail = callbacks.enqueueMail;
            callbacks.requestFile({
                .mode = FilePickerMode::Save,
                .initialPath = pickedPath,
                .extensions = {"yaml", "yml"},
                .callback = [enqueueMail, flowgraph = msg.flowgraph](std::string p) mutable {
                    enqueueMail(MailSaveFlowgraph{.flowgraph = flowgraph, .path = std::move(p)});
                },
            });

            return Result::SUCCESS;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];

        callbacks.enqueueCommand([flowgraph, path]() -> Result {
            JST_CHECK(flowgraph->exportToFile(path));
            return Result::SUCCESS;
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailCloseFlowgraph& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Cannot close flowgraph because it doesn't exist.");
            return Result::SUCCESS;
        }

        if (!msg.force && state.flowgraph.items.at(msg.flowgraph)->path().empty()) {
            state.modal.flowgraph = msg.flowgraph;
            state.modal.content = ModalContent::FlowgraphClose;
            return Result::SUCCESS;
        }

        auto inst = state.system.instance;
        auto name = msg.flowgraph;

        callbacks.enqueueCommand([inst, name]() -> Result {
            JST_CHECK(inst->flowgraphDestroy(name));
            return Result::SUCCESS;
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailSaveFlowgraphPath& msg) {
        bool validFile = true;
        if (msg.path.empty()) {
            JST_ERROR("[FLOWGRAPH] Filename is empty.");
            callbacks.notifyResult(Result::ERROR, "");
            validFile = false;
        } else {
            const std::regex filenamePattern("^.+\\.ya?ml$");
            if (!std::regex_match(msg.path, filenamePattern)) {
                JST_ERROR("[FLOWGRAPH] Invalid filename '{}'.", msg.path);
                callbacks.notifyResult(Result::ERROR, "");
                validFile = false;
            }
        }

        if (validFile) {
            callbacks.enqueueMail(MailSaveFlowgraph{.flowgraph = msg.flowgraph, .path = msg.path});
            state.modal.content.reset();
            state.modal.flowgraph.reset();
        }

        return Result::SUCCESS;
    }

    Result handle(const MailSetFlowgraphInfo& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to update flowgraph info because flowgraph was not found.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        if (msg.title.has_value()) {
            JST_CHECK(flowgraph->setTitle(*msg.title));
        }
        if (msg.summary.has_value()) {
            JST_CHECK(flowgraph->setSummary(*msg.summary));
        }
        if (msg.author.has_value()) {
            JST_CHECK(flowgraph->setAuthor(*msg.author));
        }
        if (msg.license.has_value()) {
            JST_CHECK(flowgraph->setLicense(*msg.license));
        }
        if (msg.description.has_value()) {
            JST_CHECK(flowgraph->setDescription(*msg.description));
        }

        return Result::SUCCESS;
    }

    Result handle(const MailCreateBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to create block because flowgraph was not found.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];

        std::string baseName = msg.moduleId;
        std::string blockName = baseName;
        int suffix = 1;
        while (flowgraph->view().has(blockName)) {
            blockName = jst::fmt::format("{}_{}", baseName, suffix++);
        }

        if (msg.gridPosition.has_value()) {
            const auto& pos = msg.gridPosition.value();
            const NodeMeta nodeMeta = {pos.x, pos.y, 0.0f, 0.0f};
            flowgraph->metadata().set("node", nodeMeta, blockName);
        }

        const auto moduleId = msg.moduleId;
        const auto device = msg.device;
        const auto runtime = msg.runtime;
        const auto provider = msg.provider;
        callbacks.enqueueCommand([flowgraph, blockName, moduleId, device, runtime, provider]() -> Result {
            JST_CHECK_ALLOW(flowgraph->blockCreate(blockName,
                                                  moduleId,
                                                  {},
                                                  {},
                                                  device,
                                                  runtime,
                                                  provider),
                            Result::INCOMPLETE);

            return Result::SUCCESS;
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailOpenRenameBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Cannot rename block because the flowgraph doesn't exist.");
            return Result::SUCCESS;
        }

        if (!state.flowgraph.items.at(msg.flowgraph)->view().has(msg.blockId)) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Cannot rename block because it doesn't exist.");
            return Result::SUCCESS;
        }

        state.modal.flowgraph = msg.flowgraph;
        state.modal.renameBlockOldName = msg.blockId;
        state.modal.content = ModalContent::RenameBlock;
        return Result::SUCCESS;
    }

    Result handle(const MailRenameBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Cannot rename block because the flowgraph doesn't exist.");
            return Result::SUCCESS;
        }

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        const auto oldId = msg.oldId;
        const auto newId = msg.newId;
        callbacks.enqueueCommand([flowgraph, oldId, newId]() -> Result {
            return flowgraph->blockRename(oldId, newId);
        }, false);
        callbacks.enqueueMail(MailCloseModal{});
        return Result::SUCCESS;
    }

    Result handle(const MailDeleteBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to delete block because flowgraph was not found.");
            return Result::ERROR;
        }

        if (msg.blockId.empty()) {
            JST_ERROR("Failed to delete block due to empty block ID.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        auto blockId = msg.blockId;

        callbacks.enqueueCommand([flowgraph, blockId]() -> Result {
            return flowgraph->blockDestroy(blockId);
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailReloadBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to reload block because flowgraph was not found.");
            return Result::ERROR;
        }

        if (msg.blockId.empty()) {
            JST_ERROR("Failed to reload block due to empty block ID.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        auto blockId = msg.blockId;

        callbacks.enqueueCommand([flowgraph, blockId]() -> Result {
            Parser::Map config;
            JST_CHECK(flowgraph->blockConfig(blockId, config));
            return flowgraph->blockRecreate(blockId, config);
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailChangeBlockDevice& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to change block device because flowgraph was not found.");
            return Result::ERROR;
        }

        if (msg.blockId.empty()) {
            JST_ERROR("Failed to change block device due to empty block ID.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        Flowgraph::View::BlockData block;
        if (flowgraph->view().block(msg.blockId, block) != Result::SUCCESS) {
            JST_ERROR("Failed to change block device because block was not found.");
            return Result::ERROR;
        }

        if (block.device == msg.device &&
            block.runtime == msg.runtime &&
            block.provider == msg.provider) {
            return Result::SUCCESS;
        }

        auto blockId = msg.blockId;
        auto device = msg.device;
        auto runtime = msg.runtime;
        auto provider = msg.provider;

        callbacks.enqueueCommand([flowgraph, blockId, device, runtime, provider]() -> Result {
            Parser::Map config;
            JST_CHECK(flowgraph->blockConfig(blockId, config));
            return flowgraph->blockRecreate(blockId, config, device, runtime, provider);
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailConnectBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to connect block because flowgraph was not found.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        auto blockName = msg.blockName;
        auto inputPort = msg.inputPort;
        auto sourceBlock = msg.sourceBlock;
        auto sourcePort = msg.sourcePort;

        callbacks.enqueueCommand([flowgraph, blockName, inputPort, sourceBlock, sourcePort]() -> Result {
            return flowgraph->blockConnect(blockName, inputPort, sourceBlock, sourcePort);
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailDisconnectBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to disconnect block because flowgraph was not found.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        auto blockName = msg.blockName;
        auto inputPort = msg.inputPort;

        callbacks.enqueueCommand([flowgraph, blockName, inputPort]() -> Result {
            return flowgraph->blockDisconnect(blockName, inputPort);
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailReconfigureBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to reconfigure block because flowgraph was not found.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        auto blockId = msg.blockId;
        auto config = msg.config;

        callbacks.enqueueCommand([flowgraph, blockId, config]() -> Result {
            return flowgraph->blockReconfigure(blockId, config);
        }, msg.silent);

        return Result::SUCCESS;
    }

    Result handle(const MailCopyBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to copy block because flowgraph was not found.");
            return Result::ERROR;
        }

        if (msg.blockId.empty()) {
            JST_ERROR("Failed to copy block due to empty block ID.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        Flowgraph::View::BlockData block;
        if (flowgraph->view().block(msg.blockId, block) != Result::SUCCESS) {
            JST_ERROR("Failed to copy block because block was not found.");
            return Result::ERROR;
        }

        state.clipboard.moduleType = block.type;
        state.clipboard.device = block.device;
        state.clipboard.runtime = block.runtime;
        state.clipboard.provider = block.provider;
        state.clipboard.config = block.config;
        state.clipboard.hasData = true;

        callbacks.notify(Sakura::ToastType::Info, 3000, "Block copied to clipboard.");

        return Result::SUCCESS;
    }

    Result handle(const MailPasteBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to paste block because flowgraph was not found.");
            return Result::ERROR;
        }

        if (!state.clipboard.hasData) {
            callbacks.notify(Sakura::ToastType::Warning, 3000, "Clipboard is empty.");
            return Result::SUCCESS;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];

        std::string baseName = state.clipboard.moduleType;
        std::string blockName = baseName;
        int suffix = 1;
        while (flowgraph->view().has(blockName)) {
            blockName = jst::fmt::format("{}_{}", baseName, suffix++);
        }

        if (msg.gridPosition.has_value()) {
            const auto& pos = msg.gridPosition.value();
            const NodeMeta nodeMeta = {pos.x, pos.y, 0.0f, 0.0f};
            flowgraph->metadata().set("node", nodeMeta, blockName);
        }

        auto config = state.clipboard.config;
        auto moduleType = state.clipboard.moduleType;
        auto device = state.clipboard.device;
        auto runtime = state.clipboard.runtime;
        auto provider = state.clipboard.provider;

        callbacks.enqueueCommand([flowgraph, blockName, moduleType, config, device, runtime, provider]() -> Result {
            JST_CHECK_ALLOW(flowgraph->blockCreate(blockName,
                                                  moduleType,
                                                  config,
                                                  {},
                                                  device,
                                                  runtime,
                                                  provider),
                            Result::INCOMPLETE);

            return Result::SUCCESS;
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailSetNodeMeta& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        state.flowgraph.items.at(msg.flowgraph)->metadata().set("node", msg.meta, msg.block);
        return Result::SUCCESS;
    }

    Result handle(const MailSurfaceMouse& msg) {
        if (msg.surface) {
            msg.surface->pushMouseEvent(msg.event);
        }

        return Result::SUCCESS;
    }

    Result handle(const MailResizeSurface& msg) {
        if (!msg.surface) {
            return Result::SUCCESS;
        }

        if (!msg.flowgraph.empty() && !msg.block.empty() && !msg.metaKey.empty() &&
            state.flowgraph.items.contains(msg.flowgraph)) {
            SurfaceMeta surfaceMeta;
            auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
            flowgraph->metadata().get(msg.metaKey, surfaceMeta, msg.block);
            if (msg.placement == SurfacePlacement::Attached) {
                surfaceMeta.attachedWidth = msg.resize.logicalSize.x;
                surfaceMeta.attachedHeight = msg.resize.logicalSize.y;
            } else {
                surfaceMeta.detachedWidth = msg.resize.logicalSize.x;
                surfaceMeta.detachedHeight = msg.resize.logicalSize.y;
            }
            flowgraph->metadata().set(msg.metaKey, surfaceMeta, msg.block);
        }

        SurfaceEvent event;
        event.type = SurfaceEventType::Resize;
        event.size = msg.resize.framebufferSize;
        event.scale = msg.resize.scale;
        event.backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};
        msg.surface->pushSurfaceEvent(event);

        return Result::SUCCESS;
    }

    static std::string GenerateRandomFlowgraphName() {
        static const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        std::string result = "f_";
        for (int i = 0; i < 8; ++i) {
            result += chars[rand() % chars.length()];
        }
        return result;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH
