#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH

#include "persist.hh"
#include "workbench.hh"
#include "../support/history.hh"
#include "../model/callbacks.hh"
#include "../support/clipboard.hh"
#include "../support/history.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "layout.hh"

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
#include <ranges>
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
                              MailForgetRecentFlowgraph,
                              MailRememberFlowgraph,
                              MailDropFlowgraphPath,
                              MailOpenFlowgraphBlob,
                              MailOpenFlowgraphExample,
                              MailFocusFlowgraph,
                              MailSetFlowgraphPaused,
                              MailSaveFlowgraph,
                              MailCloseFlowgraph,
                              MailCancelCloseFlowgraph,
                              MailSaveFlowgraphPath,
                              MailSetFlowgraphInfo,
                              MailCreateBlock,
                              MailOpenRenameBlock,
                              MailRenameBlock,
                              MailDeleteBlock,
                              MailToggleBlockFrozen,
                              MailReloadBlock,
                              MailChangeBlockDevice,
                              MailConnectBlock,
                              MailDisconnectBlock,
                              MailReconfigureBlock,
                              MailCopyBlock,
                              MailPasteBlock,
                              MailDuplicateBlock,
                              MailSelectBlocks,
                              MailSetNodeMeta,
                              MailSetEditorMeta,
                              MailSetConfigCollapsed,
                              MailSetMetricsCollapsed,
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

        const std::string currentPath = state.flowgraph.items.at(msg.flowgraph)->path();
        std::string path = msg.path.empty() ? currentPath : msg.path;
        if (path.empty() || msg.pick) {
            auto enqueueMail = callbacks.enqueueMail;
            callbacks.requestFile({
                .mode = FilePickerMode::Save,
                .initialPath = currentPath,
                .extensions = {"yaml", "yml"},
                .callback = [enqueueMail, flowgraph = msg.flowgraph](std::string p) mutable {
                    enqueueMail(MailSaveFlowgraph{.flowgraph = flowgraph, .path = std::move(p)});
                },
            });

            return Result::SUCCESS;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];

        JST_CHECK(LayoutActions(state.flowgraph).prepareSave(msg.flowgraph));
        bool untracked = false;
        if (const auto entry = state.flowgraph.history.find(msg.flowgraph); entry != state.flowgraph.history.end()) {
            untracked = std::exchange(entry->second.untracked, false);
        }
        auto exported = std::make_shared<FlowgraphHistory::Entry>();
        callbacks.enqueueCommandWithCompletion([flowgraph, path, exported]() -> Result {
            *exported = CaptureFlowgraphSnapshot(*flowgraph);
            JST_CHECK(flowgraph->exportToFile(path));
            return Result::SUCCESS;
        }, false, [this, name = msg.flowgraph, path, title = flowgraph->title(), exported, untracked](Result result) {
            const auto entry = state.flowgraph.history.find(name);
            if (result != Result::SUCCESS) {
                if (untracked && entry != state.flowgraph.history.end()) {
                    entry->second.untracked = true;
                }
                return;
            }
            if (entry != state.flowgraph.history.end() && *exported) {
                auto& history = entry->second;
                const auto change = history.entries.empty() ? FlowgraphChange::Content
                                                            : ClassifyFlowgraphChange(history.current(), **exported);
                history.markSaved(*exported, change == FlowgraphChange::None ||
                                                 (change == FlowgraphChange::Layout && !history.restoring));
            }
            callbacks.enqueueMail(MailRememberFlowgraph{.path = path, .title = title});
        });

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

    Result handle(const MailForgetRecentFlowgraph& msg) {
        std::erase_if(state.interface.recentFlowgraphs, [&](const auto& entry) {
            return entry.path == msg.path;
        });
        return PersistSettings([&](Settings& settings) {
            settings.interface.recentFlowgraphs = state.interface.recentFlowgraphs;
        });
    }

    Result handle(const MailCloseFlowgraph& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Cannot close flowgraph because it doesn't exist.");
            return Result::SUCCESS;
        }

        auto& ui = state.interface;
        if (!msg.force && state.flowgraphMayLoseChanges(msg.flowgraph)) {
            ui.closingFlowgraph = msg.flowgraph;
            ui.quitting = false;
            return Result::SUCCESS;
        }
        if (ui.quitting && ui.closingFlowgraph == msg.flowgraph) {
            ui.discardedFlowgraphs.insert(msg.flowgraph);
            return WorkbenchActions::Quit(state);
        }
        ui.closingFlowgraph.reset();

        auto inst = state.system.instance;
        auto name = msg.flowgraph;

        callbacks.enqueueCommand([inst, name]() -> Result {
            JST_CHECK(inst->flowgraphDestroy(name));
            return Result::SUCCESS;
        }, false);

        return Result::SUCCESS;
    }

    Result handle(const MailCancelCloseFlowgraph&) {
        state.interface.closingFlowgraph.reset();
        state.interface.quitting = false;
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

    Result handle(const MailSetFlowgraphPaused& msg) {
        const auto entry = state.flowgraph.items.find(msg.flowgraph);
        if (entry == state.flowgraph.items.end()) {
            return Result::SUCCESS;
        }

        const auto& flowgraph = entry->second;
        const Result result = msg.paused ? flowgraph->pause() : flowgraph->resume();
        if (result != Result::SUCCESS) {
            callbacks.notifyResult(result, msg.paused ? "Cannot pause the flowgraph." : "Cannot resume the flowgraph.");
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
            if (const auto layout = state.flowgraph.layouts.find(msg.flowgraph); layout != state.flowgraph.layouts.end()) {
                JST_CHECK(LayoutActions(state.flowgraph).handle(MailSetFlowgraphLayout{msg.flowgraph, layout->second}));
            }
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
        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, blockName, moduleId, device, runtime, provider]() -> Result {
                JST_CHECK_ALLOW(flowgraph->blockCreate(blockName,
                                                      moduleId,
                                                      {},
                                                      {},
                                                      device,
                                                      runtime,
                                                      provider),
                                Result::INCOMPLETE);

                return Result::SUCCESS;
            },
        });

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
        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, oldId, newId]() -> Result {
                return flowgraph->blockRename(oldId, newId);
            },
            .onComplete = mailOnSuccess(MailRenameLayoutBlock{
                .flowgraph = msg.flowgraph,
                .oldId = oldId,
                .newId = newId,
            }),
        });
        callbacks.enqueueMail(MailCloseModal{});
        return Result::SUCCESS;
    }

    Result handle(const MailDeleteBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to delete block because flowgraph was not found.");
            return Result::ERROR;
        }

        std::vector<std::string> blockIds;
        for (const auto& blockId : msg.blockIds) {
            if (!blockId.empty() && std::find(blockIds.begin(), blockIds.end(), blockId) == blockIds.end()) {
                blockIds.push_back(blockId);
            }
        }
        if (blockIds.empty()) {
            JST_ERROR("Failed to delete blocks due to empty block IDs.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];
        auto destroyed = std::make_shared<std::vector<std::string>>();

        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, blockIds, destroyed]() -> Result {
                for (const auto& blockId : blockIds) {
                    JST_CHECK(flowgraph->blockDestroy(blockId));
                    destroyed->push_back(blockId);
                }
                return Result::SUCCESS;
            },
            .onComplete = [enqueueMail = callbacks.enqueueMail, flowgraphId = msg.flowgraph, destroyed](Result) {
                for (const auto& blockId : *destroyed) {
                    enqueueMail(MailRemoveLayoutBlock{.flowgraph = flowgraphId, .block = blockId});
                }
            },
        });

        return Result::SUCCESS;
    }

    Result handle(const MailToggleBlockFrozen& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to toggle block freeze because flowgraph was not found.");
            return Result::ERROR;
        }

        std::vector<std::string> blockIds;
        for (const auto& blockId : msg.blockIds) {
            if (!blockId.empty() && std::find(blockIds.begin(), blockIds.end(), blockId) == blockIds.end()) {
                blockIds.push_back(blockId);
            }
        }
        if (blockIds.empty()) {
            JST_ERROR("Failed to toggle block freeze due to empty block IDs.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items[msg.flowgraph];

        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, blockIds]() -> Result {
                bool freeze = false;
                for (const auto& blockId : blockIds) {
                    Flowgraph::View::BlockInfo block;
                    if (flowgraph->view().info(blockId, block) == Result::SUCCESS && !block.frozen) {
                        freeze = true;
                    }
                }

                for (const auto& blockId : blockIds) {
                    JST_CHECK(flowgraph->blockFreeze(blockId, freeze));
                }
                return Result::SUCCESS;
            },
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

        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, blockId, device, runtime, provider]() -> Result {
                Parser::Map config;
                JST_CHECK(flowgraph->blockConfig(blockId, config));
                return flowgraph->blockRecreate(blockId, config, device, runtime, provider);
            },
        });

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

        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, blockName, inputPort, sourceBlock, sourcePort]() -> Result {
                return flowgraph->blockConnect(blockName, inputPort, sourceBlock, sourcePort);
            },
        });

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

        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, blockName, inputPort]() -> Result {
                return flowgraph->blockDisconnect(blockName, inputPort);
            },
        });

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

        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .mergeKey = ReconfigureMergeKey(blockId, config),
            .command = [flowgraph, blockId, config]() -> Result {
                return flowgraph->blockReconfigure(blockId, config);
            },
            .silent = msg.silent,
            .onComplete = msg.onComplete,
        });

        return Result::SUCCESS;
    }

    Result handle(const MailCopyBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to copy blocks because flowgraph was not found.");
            return Result::ERROR;
        }

        if (msg.blockIds.empty()) {
            JST_ERROR("Failed to copy blocks due to empty block IDs.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        auto copied = std::make_shared<FlowgraphSnapshot>();
        auto destroyed = std::make_shared<std::vector<std::string>>();
        auto yaml = std::make_shared<std::string>();
        auto previous = std::exchange(pendingCopy, yaml);
        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, blockIds = msg.blockIds, cut = msg.cut, copied, destroyed, yaml,
                        previous]() -> Result {
                if (previous) {
                    *yaml = *previous;
                }
                const auto snapshot = CaptureFlowgraphSnapshot(*flowgraph);
                if (!snapshot) {
                    return Result::ERROR;
                }
                *copied = CopyFlowgraphBlocks(*snapshot, blockIds);
                if (copied->blocks.empty()) {
                    return Result::SUCCESS;
                }
                JST_CHECK(EncodeFlowgraphBlocks(*copied, *yaml));
                if (cut) {
                    const auto order = FlowgraphDependencyOrder(*copied);
                    for (const auto* block : order | std::views::reverse) {
                        JST_CHECK(flowgraph->blockDestroy(block->name));
                        destroyed->push_back(block->name);
                    }
                }
                return Result::SUCCESS;
            },
            .silent = !msg.cut,
            .onComplete = [this, flowgraphId = msg.flowgraph, cut = msg.cut, copied, destroyed, yaml](Result result) {
                if (pendingCopy == yaml) {
                    pendingCopy.reset();
                }
                if (copied->blocks.empty()) {
                    callbacks.notify(Sakura::ToastType::Error, 5000, "Cannot copy blocks because they no longer exist.");
                    return;
                }

                if (!yaml->empty()) {
                    callbacks.setClipboardText(*yaml);
                }
                for (const auto& block : *destroyed) {
                    callbacks.enqueueMail(MailRemoveLayoutBlock{.flowgraph = flowgraphId, .block = block});
                }
                if (result != Result::SUCCESS) {
                    return;
                }

                const U64 count = copied->blocks.size();
                callbacks.notify(Sakura::ToastType::Info,
                                 3000,
                                 jst::fmt::format("{} {} to clipboard.",
                                                  count == 1 ? std::string("Block") : jst::fmt::format("{} blocks", count),
                                                  cut ? "cut" : "copied"));
            },
        });

        return Result::SUCCESS;
    }

    Result handle(const MailPasteBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to paste blocks because flowgraph was not found.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        auto created = std::make_shared<std::vector<std::string>>();
        auto empty = std::make_shared<bool>(false);
        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, pending = pendingCopy, text = callbacks.getClipboardText(),
                        point = msg.gridPosition, created, empty]() -> Result {
                const auto blocks = DecodeFlowgraphBlocks(pending && !pending->empty() ? *pending : text);
                if (!blocks.has_value()) {
                    *empty = true;
                    return Result::SUCCESS;
                }
                return PasteFlowgraphBlocks(*flowgraph, *blocks, point, *created);
            },
            .onComplete = [this, select = selectOnComplete(msg.flowgraph, created), empty](Result result) {
                if (*empty) {
                    callbacks.notify(Sakura::ToastType::Warning, 3000, "Clipboard has no blocks to paste.");
                }
                select(result);
            },
        });

        return Result::SUCCESS;
    }

    Result handle(const MailDuplicateBlock& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to duplicate blocks because flowgraph was not found.");
            return Result::ERROR;
        }

        if (msg.blockIds.empty()) {
            JST_ERROR("Failed to duplicate blocks due to empty block IDs.");
            return Result::ERROR;
        }

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        auto created = std::make_shared<std::vector<std::string>>();
        callbacks.enqueueFlowgraphEdit({
            .flowgraph = msg.flowgraph,
            .command = [flowgraph, blockIds = msg.blockIds, created]() -> Result {
                const auto snapshot = CaptureFlowgraphSnapshot(*flowgraph);
                if (!snapshot) {
                    return Result::ERROR;
                }
                return PasteFlowgraphBlocks(*flowgraph, CopyFlowgraphBlocks(*snapshot, blockIds), {}, *created);
            },
            .onComplete = selectOnComplete(msg.flowgraph, created),
        });

        return Result::SUCCESS;
    }

    Result handle(const MailSelectBlocks& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto& selection = state.flowgraph.selections[msg.flowgraph];
        selection.requests += 1;
        selection.blocks = msg.blockIds;
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
        if (const auto history = state.flowgraph.history.find(msg.flowgraph);
            history != state.flowgraph.history.end()) {
            history->second.place(msg.block, merged);
        }
        return Result::SUCCESS;
    }

    Result handle(const MailSetEditorMeta& msg) {
        if (const auto graph = state.flowgraph.items.find(msg.flowgraph); graph != state.flowgraph.items.end()) {
            return graph->second->metadata().set("editor", msg.meta);
        }
        return Result::SUCCESS;
    }

    Result handle(const MailSetConfigCollapsed& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        ConfigMeta existing;
        flowgraph->metadata().get("config", existing, msg.block);
        if (existing.collapsed != msg.collapsed) {
            markUntracked(msg.flowgraph);
        }
        existing.collapsed = msg.collapsed;
        flowgraph->metadata().set("config", existing, msg.block);
        return Result::SUCCESS;
    }

    Result handle(const MailSetMetricsCollapsed& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        ConfigMeta existing;
        flowgraph->metadata().get("config", existing, msg.block);
        if (existing.tileMetricsCollapsed != msg.collapsed) {
            markUntracked(msg.flowgraph);
        }
        existing.tileMetricsCollapsed = msg.collapsed;
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

 private:
    std::shared_ptr<std::string> pendingCopy;

    void markUntracked(const std::string& flowgraph) {
        if (const auto history = state.flowgraph.history.find(flowgraph); history != state.flowgraph.history.end()) {
            history->second.untracked = true;
        }
    }

    static std::string ReconfigureMergeKey(const std::string& block, const Parser::Map& config) {
        std::vector<std::string> keys;
        for (const auto& [key, _] : config) {
            keys.push_back(key);
        }
        std::sort(keys.begin(), keys.end());
        std::string mergeKey = block;
        for (const auto& key : keys) {
            mergeKey += "\n" + key;
        }
        return mergeKey;
    }

    Sakura::Callback<void(Result)> selectOnComplete(const std::string& flowgraphId,
                                                    std::shared_ptr<std::vector<std::string>> created) {
        return [enqueueMail = callbacks.enqueueMail, flowgraphId, created](Result) {
            if (!created->empty()) {
                enqueueMail(MailSelectBlocks{.flowgraph = flowgraphId, .blockIds = std::move(*created)});
            }
        };
    }

    Sakura::Callback<void(Result)> mailOnSuccess(Mail mail) const {
        return [enqueueMail = callbacks.enqueueMail, mail = std::move(mail)](Result result) {
            if (result == Result::SUCCESS) {
                enqueueMail(Mail(mail));
            }
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FLOWGRAPH_HH
