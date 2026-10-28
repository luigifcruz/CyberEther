#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH

#include "persist.hh"
#include "workbench.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../support/gallery.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/instance.hh"
#include "jetstream/platform.hh"
#include "jetstream/registry.hh"
#include "tools/naming.hh"
#include "tools/text.hh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphActions {
    using Filter = std::tuple<MailNewFlowgraph,
                              MailOpenFlowgraph,
                              MailOpenFlowgraphPath,
                              MailToggleFavoriteFlowgraph,
                              MailRememberFlowgraph,
                              MailDropFlowgraphPath,
                              MailOpenFlowgraphBlob,
                              MailOpenFlowgraphExample,
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
                              MailSetConfigCollapsed,
                              MailSurfaceInput,
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
        auto title = std::make_shared<std::string>();
        auto enqueueMail = callbacks.enqueueMail;

        callbacks.enqueueCommandWithCompletion([inst, name, path, title]() -> Result {
            std::shared_ptr<Flowgraph> flowgraph;
            JST_CHECK(inst->flowgraphCreate(name, {}, flowgraph));
            JST_CHECK(flowgraph->importFromFile(path));
            *title = flowgraph->title();
            return Result::SUCCESS;
        }, false, [enqueueMail, path, title](Result result) {
            if (result == Result::SUCCESS) {
                enqueueMail(MailRememberFlowgraph{.path = path, .title = *title});
            }
        });

        state.interface.pendingFocusedFlowgraph = name;
        return Result::SUCCESS;
    }

    Result handle(const MailDropFlowgraphPath& msg) {
        const auto extension = Text::ToLower(Platform::PathToUtf8(
            Platform::PathFromUtf8(msg.path).extension()));

        if (extension != ".yml" && extension != ".yaml") {
            callbacks.notify(Sakura::ToastType::Warning, 5000,
                             "Only flowgraph files (.yml or .yaml) can be opened by dropping them here.");
            return Result::SUCCESS;
        }

        return handle(MailOpenFlowgraphPath{msg.path});
    }

    Result handle(const MailOpenFlowgraphBlob& msg) {
        return openBlob(msg.blob).has_value() ? Result::SUCCESS : Result::ERROR;
    }

    std::optional<std::string> openBlob(std::vector<char> blob) {
        if (blob.empty()) {
            JST_ERROR("Failed to open flowgraph from blob due to invalid blob data.");
            return std::nullopt;
        }

        auto inst = state.system.instance;
        auto name = GenerateRandomFlowgraphName();

        callbacks.enqueueCommandWithCompletion([inst, name, blob = std::move(blob)]() -> Result {
            std::shared_ptr<Flowgraph> flowgraph;
            JST_CHECK(inst->flowgraphCreate(name, {}, flowgraph));
            JST_CHECK(flowgraph->importFromBlob(blob));
            return Result::SUCCESS;
        }, false, [this, name](Result result) {
            if (result != Result::SUCCESS) {
                WorkbenchActions::Forget(state, name);
            }
        });

        state.interface.pendingFocusedFlowgraph = name;
        return name;
    }

    static std::string GalleryDescription(const Registry::FlowgraphRegistration& entry) {
        static const std::string placeholder = Registry::FlowgraphRegistration{}.description;
        if (entry.description == placeholder) {
            return {};
        }
        std::string markdown = entry.description;
        const std::string_view host = Gallery::Host;
        for (U64 at = markdown.find("](/"); at != std::string::npos; at = markdown.find("](/", at)) {
            markdown.insert(at + 2, host);
            at += 2 + host.size();
        }
        return markdown;
    }

    Result handle(const MailOpenFlowgraphExample& msg) {
        const auto& flowgraphs = state.registry.flowgraphs;
        const auto example = std::find_if(flowgraphs.begin(), flowgraphs.end(), [&](const auto& entry) {
            return entry.key == msg.key;
        });
        if (example == flowgraphs.end()) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "The selected example is no longer available.");
            return Result::SUCCESS;
        }
        const auto name = openBlob({example->content.begin(), example->content.end()});
        if (!name.has_value()) {
            return Result::ERROR;
        }
        state.flowgraph.gallery[*name] = GalleryDescription(*example);
        return Result::SUCCESS;
    }

    Result handle(const MailFocusFlowgraph& msg) {
        if (state.flowgraph.items.contains(msg.flowgraph)) {
            state.interface.focusedFlowgraph = msg.flowgraph;
            WorkbenchActions::Focus(state, std::nullopt);
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

        callbacks.enqueueCommandWithMail([flowgraph, path]() -> Result {
            JST_CHECK(flowgraph->exportToFile(path));
            return Result::SUCCESS;
        }, false, MailRememberFlowgraph{.path = path, .title = flowgraph->title()});

        return Result::SUCCESS;
    }

    static std::string TitleFromFile(const std::string& path) {
        std::string title;
        if (Flowgraph::ReadTitle(path, title) != Result::SUCCESS) {
            return "";
        }
        return title;
    }

    static void Remember(DefaultCompositorState::InterfaceState& ui, const std::string& path, std::string title,
                         U64 accessed) {
        auto& recent = ui.recentFlowgraphs;
        std::erase_if(recent, [&](const auto& entry) {
            return entry.path == path;
        });
        recent.insert(recent.begin(), {.path = path, .title = std::move(title), .accessed = accessed});
        if (recent.size() > DefaultCompositorState::InterfaceState::MaxRecentFlowgraphs) {
            recent.resize(DefaultCompositorState::InterfaceState::MaxRecentFlowgraphs);
        }
        for (auto& favorite : ui.favoriteFlowgraphs) {
            if (favorite.path == path) {
                favorite = recent.front();
            }
        }
    }

    static void ToggleFavorite(DefaultCompositorState::InterfaceState& ui, const std::string& path) {
        auto& favorites = ui.favoriteFlowgraphs;
        if (std::erase_if(favorites, [&](const auto& entry) { return entry.path == path; }) > 0) {
            return;
        }
        const auto& recent = ui.recentFlowgraphs;
        const auto entry = std::find_if(recent.begin(), recent.end(), [&](const auto& item) {
            return item.path == path;
        });
        favorites.push_back(entry != recent.end() ? *entry
                                                  : Settings::Interface::FlowgraphEntry{
                                                        .path = path,
                                                        .title = TitleFromFile(path),
                                                    });
    }

    Result handle(const MailRememberFlowgraph& msg) {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        Remember(state.interface, msg.path, msg.title,
                 static_cast<U64>(std::chrono::duration_cast<std::chrono::seconds>(now).count()));
        const auto result = PersistSettings([&](Settings& settings) {
            settings.interface.recentFlowgraphs = state.interface.recentFlowgraphs;
            settings.interface.favoriteFlowgraphs = state.interface.favoriteFlowgraphs;
        });
        if (result != Result::SUCCESS) {
            JST_WARN("[COMPOSITOR] Failed to save the recent flowgraph list.");
        }
        return Result::SUCCESS;
    }

    Result handle(const MailToggleFavoriteFlowgraph& msg) {
        ToggleFavorite(state.interface, msg.path);
        return PersistSettings([&](Settings& settings) {
            settings.interface.favoriteFlowgraphs = state.interface.favoriteFlowgraphs;
        });
    }

    Result handle(const MailCloseFlowgraph& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Cannot close flowgraph because it doesn't exist.");
            return Result::SUCCESS;
        }

        const auto& flowgraph = state.flowgraph.items.at(msg.flowgraph);
        if (!msg.force && !flowgraph->view().empty()) {
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

        const std::string blockName = Naming::UniqueName(msg.moduleId, [&](const std::string& candidate) {
            return flowgraph->view().has(candidate);
        });

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
        callbacks.enqueueCommandWithMail([flowgraph, oldId, newId]() -> Result {
            return flowgraph->blockRename(oldId, newId);
        }, false, MailRenameLayoutBlock{
            .flowgraph = msg.flowgraph,
            .oldId = oldId,
            .newId = newId,
        });
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

        callbacks.enqueueCommandWithMail([flowgraph, blockId]() -> Result {
            return flowgraph->blockDestroy(blockId);
        }, false, MailRemoveLayoutBlock{
            .flowgraph = msg.flowgraph,
            .block = blockId,
        });

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
            if (msg.onComplete) {
                msg.onComplete(Result::ERROR);
            }
            JST_ERROR("Failed to reconfigure block because flowgraph was not found.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        auto blockId = msg.blockId;
        auto config = msg.config;

        auto command = [flowgraph, blockId, config]() -> Result {
            return flowgraph->blockReconfigure(blockId, config);
        };
        if (msg.onComplete) {
            callbacks.enqueueCommandWithCompletion(std::move(command), msg.silent, msg.onComplete);
        } else {
            callbacks.enqueueCommand(std::move(command), msg.silent);
        }

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

        const std::string blockName = Naming::UniqueName(state.clipboard.moduleType, [&](const std::string& candidate) {
            return flowgraph->view().has(candidate);
        });

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

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        NodeMeta existing;
        flowgraph->metadata().get("node", existing, msg.block);
        NodeMeta merged = msg.meta;
        if (msg.base.has_value()) {
            for (const auto field : {&NodeMeta::x, &NodeMeta::y, &NodeMeta::width, &NodeMeta::height}) {
                if (existing.*field != (*msg.base).*field) {
                    merged.*field = existing.*field;
                }
            }
        }
        flowgraph->metadata().set("node", merged, msg.block);
        return Result::SUCCESS;
    }

    Result handle(const MailSetConfigCollapsed& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        ConfigMeta existing;
        flowgraph->metadata().get("config", existing, msg.block);
        existing.collapsed = msg.collapsed;
        flowgraph->metadata().set("config", existing, msg.block);
        return Result::SUCCESS;
    }

    Result handle(const MailSurfaceInput& msg) {
        if (msg.surface) {
            msg.surface->pushInputEvent(msg.event);
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
            if (msg.placement == SurfacePlacement::Attached && surfaceMeta.attachedHeight != msg.resize.logicalSize.y) {
                surfaceMeta.attachedHeight = msg.resize.logicalSize.y;
                flowgraph->metadata().set(msg.metaKey, surfaceMeta, msg.block);
            }
        }

        SurfaceEvent event;
        event.type = SurfaceEventType::Resize;
        event.size = msg.resize.framebufferSize;
        event.scale = msg.resize.scale;
        event.placement = (msg.placement == SurfacePlacement::Attached)
                              ? SurfacePlacementType::Attached
                              : SurfacePlacementType::Detached;
        event.theme = Sakura::MakeSurfaceTheme(state.sakura.colorMap, event.placement);
        msg.surface->pushSurfaceEvent(event);
        state.flowgraph.surfaces[msg.surface.get()] = {.surface = msg.surface, .event = event};

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
