#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "jetstream/render/sakura/sakura.hh"
#include "../model/state.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
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
                              MailSaveFlowgraph,
                              MailCloseFlowgraph,
                              MailSaveFlowgraphPath,
                              MailBrowseConfigPath,
                              MailSetFlowgraphInfo,
                              MailCreateBlock,
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
        Platform::PickFile(path, {"yaml", "yml"}, [enqueueMail](std::string p) mutable {
            enqueueMail(MailOpenFlowgraphPath{std::move(p)});
        });

        return Result::SUCCESS;
    }

    Result handle(const MailOpenFlowgraphPath& msg) {
        if (msg.path.empty()) {
            Sakura::Notify(Sakura::NotificationType::Error, 5000, "Cannot open flowgraph due to empty path.");
            return Result::SUCCESS;
        }

        if (!std::filesystem::exists(msg.path)) {
            Sakura::Notify(Sakura::NotificationType::Error, 5000, "The selected file does not exist.");
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

    Result handle(const MailSaveFlowgraph& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            Sakura::Notify(Sakura::NotificationType::Error,
                           5000,
                           "Cannot save flowgraph because it doesn't exist.");
            return Result::SUCCESS;
        }

        std::string path = msg.path.empty() ? state.flowgraph.items.at(msg.flowgraph)->path() : msg.path;
        if (path.empty()) {
            std::string pickedPath;
            auto enqueueMail = callbacks.enqueueMail;
            Platform::SaveFile(pickedPath, [enqueueMail, flowgraph = msg.flowgraph](std::string p) mutable {
                enqueueMail(MailSaveFlowgraph{.flowgraph = flowgraph, .path = std::move(p)});
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
            Sakura::Notify(Sakura::NotificationType::Error,
                           5000,
                           "Cannot close flowgraph because it doesn't exist.");
            return Result::SUCCESS;
        }

        if (!msg.force && state.flowgraph.items.at(msg.flowgraph)->path().empty()) {
            state.modal.flowgraph = msg.flowgraph;
            state.modal.content = DefaultCompositorState::ModalState::Content::FlowgraphClose;
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
            Sakura::NotifyResultClean(Result::ERROR);
            validFile = false;
        } else {
            const std::regex filenamePattern("^.+\\.ya?ml$");
            if (!std::regex_match(msg.path, filenamePattern)) {
                JST_ERROR("[FLOWGRAPH] Invalid filename '{}'.", msg.path);
                Sakura::NotifyResultClean(Result::ERROR);
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

    Result handle(const MailBrowseConfigPath& msg) {
        std::string path = msg.path;
        const auto callback = [onSelect = msg.onSelect](std::string selectedPath) {
            if (onSelect) {
                onSelect(std::move(selectedPath));
            }
        };

        const Result result = msg.save
            ? Platform::SaveFile(path, callback)
            : Platform::PickFile(path, msg.extensions, callback);
        if (result != Result::SUCCESS && !Platform::IsFilePending()) {
            Sakura::NotifyResultClean(result);
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
        const auto existingBlocks = flowgraph->blockList();

        std::string baseName = msg.moduleId;
        std::string blockName = baseName;
        int suffix = 1;
        while (existingBlocks.contains(blockName)) {
            blockName = jst::fmt::format("{}_{}", baseName, suffix++);
        }

        if (msg.gridPosition.has_value()) {
            const auto& pos = msg.gridPosition.value();
            const NodeMeta nodeMeta = {pos.x, pos.y, 140.0f, 0.0f};
            flowgraph->setMeta("node", nodeMeta, blockName);
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

    Result handle(const MailRenameBlock&) {
        // TODO: Implement.
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
        const auto blocks = flowgraph->blockList();
        if (!blocks.contains(msg.blockId)) {
            JST_ERROR("Failed to change block device because block was not found.");
            return Result::ERROR;
        }

        const auto& block = blocks.at(msg.blockId);
        if (block->device() == msg.device &&
            block->runtime() == msg.runtime &&
            block->provider() == msg.provider) {
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
        const auto blocks = flowgraph->blockList();
        if (!blocks.contains(msg.blockId)) {
            JST_ERROR("Failed to copy block because block was not found.");
            return Result::ERROR;
        }

        const auto& block = blocks.at(msg.blockId);

        state.clipboard.moduleType = block->config().type();
        state.clipboard.device = block->device();
        state.clipboard.runtime = block->runtime();
        state.clipboard.provider = block->provider();
        flowgraph->blockConfig(msg.blockId, state.clipboard.config);
        state.clipboard.hasData = true;

        Sakura::Notify(Sakura::NotificationType::Info, 3000, "Block copied to clipboard.");

        return Result::SUCCESS;
    }

    Result handle(const MailPasteBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to paste block because flowgraph was not found.");
            return Result::ERROR;
        }

        if (!state.clipboard.hasData) {
            Sakura::Notify(Sakura::NotificationType::Warning, 3000, "Clipboard is empty.");
            return Result::SUCCESS;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        const auto existingBlocks = flowgraph->blockList();

        std::string baseName = state.clipboard.moduleType;
        std::string blockName = baseName;
        int suffix = 1;
        while (existingBlocks.contains(blockName)) {
            blockName = jst::fmt::format("{}_{}", baseName, suffix++);
        }

        if (msg.gridPosition.has_value()) {
            const auto& pos = msg.gridPosition.value();
            const NodeMeta nodeMeta = {pos.x, pos.y, 140.0f, 0.0f};
            flowgraph->setMeta("node", nodeMeta, blockName);
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

        state.flowgraph.items.at(msg.flowgraph)->setMeta("node", msg.meta, msg.block);
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
            (msg.attached.has_value() || msg.detached.has_value()) && state.flowgraph.items.contains(msg.flowgraph)) {
            SurfaceMeta surfaceMeta;
            auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
            flowgraph->getMeta(msg.metaKey, surfaceMeta, msg.block);
            if (msg.attached.has_value()) {
                surfaceMeta.attachedWidth = msg.attached->x;
                surfaceMeta.attachedHeight = msg.attached->y;
            }
            if (msg.detached.has_value()) {
                surfaceMeta.detachedWidth = msg.detached->x;
                surfaceMeta.detachedHeight = msg.detached->y;
            }
            flowgraph->setMeta(msg.metaKey, surfaceMeta, msg.block);
        }

        const auto& bg = state.sakura.colorMap.at(msg.detachedSurface ? "background" : "node_background");
        SurfaceEvent event;
        event.type = SurfaceEventType::Resize;
        event.size = {msg.width, msg.height};
        event.scale = msg.scale;
        event.backgroundColor = {bg.x, bg.y, bg.z, bg.w};
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
