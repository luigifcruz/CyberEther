#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH

#include "gallery.hh"
#include "persist.hh"
#include "settings.hh"
#include "../support/docs.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "render/sakura/themes.hh"

#include "jetstream/settings.hh"
#include "jetstream/platform.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>

namespace Jetstream {

struct WorkbenchActions {
    using Filter = std::tuple<MailApplyTheme,
                              MailOpenModal,
                              MailCloseModal,
                              MailNotify,
                              MailNotifyResult,
                              MailOpenUrl,
                              MailCopyText,
                              MailQuit,
                              MailOpenStartTab,
                              MailCloseStartTab,
                              MailFocusStartTab,
                              MailSetWelcomeSection,
                              MailSetWelcomeQuery,
                              MailSetWelcomeCategory,
                              MailSetWelcomePage,
                              MailOpenSettings,
                              MailCloseSettings,
                              MailFocusSettings,
                              MailOpenDocs,
                              MailCloseDocs,
                              MailFocusDocs>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    WorkbenchActions(DefaultCompositorState& state,
                     DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    static void Focus(DefaultCompositorState& state, std::optional<WorkspaceTab> tab) {
        state.interface.focusedTab = tab;
        state.interface.tabNavigationPending = true;
    }

    static void Open(DefaultCompositorState& state, WorkspaceTab tab) {
        auto& ui = state.interface;
        switch (tab) {
            case WorkspaceTab::Start:
                if (!ui.startTab) {
                    ui.startTab = true;
                    ui.startTabBaseline = state.flowgraph.items.size();
                }
                ++ui.startTabRequests;
                break;
            case WorkspaceTab::Settings:
                ui.settingsTab = true;
                ++ui.settingsTabRequests;
                break;
            case WorkspaceTab::Docs:
                ui.docsTab = true;
                ++ui.docsTabRequests;
                break;
        }
        Focus(state, tab);
    }

    static void Close(DefaultCompositorState& state, WorkspaceTab tab) {
        auto& ui = state.interface;
        switch (tab) {
            case WorkspaceTab::Start:
                ui.startTab = false;
                break;
            case WorkspaceTab::Settings:
                ui.settingsTab = false;
                SettingsActions::ResetPages(state);
                break;
            case WorkspaceTab::Docs:
                ui.docsTab = false;
                break;
        }
        if (ui.focusedTab == tab && !state.tabShown(tab)) {
            Focus(state, std::nullopt);
        }
    }

    static void OpenSettings(DefaultCompositorState& state, SettingsSection section) {
        SettingsActions::Select(state, section);
        Open(state, WorkspaceTab::Settings);
    }

    Result handle(const MailOpenStartTab&) {
        Open(state, WorkspaceTab::Start);
        return Result::SUCCESS;
    }

    Result handle(const MailCloseStartTab&) {
        Close(state, WorkspaceTab::Start);
        return Result::SUCCESS;
    }

    Result handle(const MailSetWelcomeSection& msg) {
        state.interface.welcomeSection = msg.section;
        state.interface.welcomePage = 0;
        return Result::SUCCESS;
    }

    Result handle(const MailSetWelcomeQuery& msg) {
        state.interface.welcomeQuery = msg.query;
        state.interface.welcomePage = 0;
        return Result::SUCCESS;
    }

    Result handle(const MailSetWelcomeCategory& msg) {
        state.interface.welcomeCategory = msg.category;
        state.interface.welcomePage = 0;
        return Result::SUCCESS;
    }

    Result handle(const MailSetWelcomePage& msg) {
        state.interface.welcomePage = msg.page;
        return Result::SUCCESS;
    }

    Result handle(const MailFocusStartTab&) {
        focusShown(WorkspaceTab::Start);
        return Result::SUCCESS;
    }

    Result handle(const MailOpenSettings& msg) {
        OpenSettings(state, msg.section);
        return Result::SUCCESS;
    }

    Result handle(const MailCloseSettings&) {
        Close(state, WorkspaceTab::Settings);
        return Result::SUCCESS;
    }

    Result handle(const MailFocusSettings&) {
        focusShown(WorkspaceTab::Settings);
        return Result::SUCCESS;
    }

    Result handle(const MailOpenDocs& msg) {
        auto& ui = state.interface;
        if (!msg.page.empty() || !ui.docsTab) {
            ui.docsPage = Docs::Bundled().find(msg.page).has_value() ? msg.page : std::string(Docs::Home);
            ui.docsAnchor = msg.anchor;
            ++ui.docsNavigations;
        }
        Open(state, WorkspaceTab::Docs);
        return Result::SUCCESS;
    }

    Result handle(const MailCloseDocs&) {
        Close(state, WorkspaceTab::Docs);
        return Result::SUCCESS;
    }

    Result handle(const MailFocusDocs&) {
        focusShown(WorkspaceTab::Docs);
        return Result::SUCCESS;
    }

    void settlePendingFocus() {
        auto& ui = state.interface;
        if (!ui.pendingFocusedFlowgraph || !state.flowgraph.items.contains(*ui.pendingFocusedFlowgraph)) {
            return;
        }
        ui.focusedFlowgraph = ui.pendingFocusedFlowgraph;
        ui.pendingFocusedFlowgraph.reset();
        Focus(state, std::nullopt);
    }

    void reconcileFlowgraphs() {
        auto& ui = state.interface;
        const auto& items = state.flowgraph.items;
        for (const auto& name : knownFlowgraphs) {
            if (!items.contains(name)) {
                Forget(state, name);
            }
        }
        knownFlowgraphs.clear();
        for (const auto& [name, _] : items) {
            knownFlowgraphs.insert(name);
        }

        if (ui.focusedFlowgraph.has_value() && !items.contains(*ui.focusedFlowgraph)) {
            ui.focusedFlowgraph.reset();
        }
        if (ui.closingFlowgraph.has_value() && !items.contains(*ui.closingFlowgraph)) {
            ui.closingFlowgraph.reset();
        }
        if (!ui.focusedFlowgraph.has_value() && !ui.pendingFocusedFlowgraph.has_value() && items.size() == 1) {
            ui.focusedFlowgraph = items.begin()->first;
        }
    }

    static void Forget(DefaultCompositorState& state, const std::string& flowgraph) {
        state.flowgraph.gallery.erase(flowgraph);
        state.flowgraph.environment.erase(flowgraph);
        if (state.interface.pendingFocusedFlowgraph == flowgraph) {
            state.interface.pendingFocusedFlowgraph.reset();
        }
    }

    void reconcileModal() {
        auto& modal = state.modal;
        if (!modal.content.has_value()) {
            modal.flowgraph.reset();
            return;
        }

        if (modal.content == ModalContent::RemoteQr && !state.remote.started) {
            modal.content.reset();
            return;
        }

        if (modal.content != ModalContent::FlowgraphInfo && modal.content != ModalContent::RenameBlock) {
            return;
        }

        const auto target = modal.flowgraph.has_value() ? modal.flowgraph : state.interface.focusedFlowgraph;
        const bool missing = !target.has_value() || !state.flowgraph.items.contains(target.value());
        const bool renameMissing = modal.content == ModalContent::RenameBlock &&
                                   (missing || !modal.renameBlockOldName.has_value() ||
                                    !state.flowgraph.items.at(target.value())->view().has(
                                        modal.renameBlockOldName.value()));
        if (missing || renameMissing) {
            modal.content.reset();
            modal.flowgraph.reset();
            modal.renameBlockOldName.reset();
        }
    }

    void settleStartTab() {
        auto& ui = state.interface;
        const bool navigated = ui.tabNavigationPending;
        const U64 count = state.flowgraph.items.size();
        if (ui.startTab && (count == 0 || count > ui.startTabBaseline)) {
            Close(state, WorkspaceTab::Start);
        } else if (ui.startTab) {
            ui.startTabBaseline = count;
        }

        const bool shown = state.startTabShown();
        if (shown && !startTabWasShown && !navigated) {
            Focus(state, WorkspaceTab::Start);
            ++ui.startTabRequests;
        }
        startTabWasShown = shown;
        if (ui.focusedTab.has_value() && !state.tabShown(*ui.focusedTab)) {
            Focus(state, std::nullopt);
        }
        ui.tabNavigationPending = false;
    }

    Result handle(const MailApplyTheme& msg) {
        if (!Sakura::Themes.contains(msg.themeKey)) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Cannot apply theme because it doesn't exist.");
            return Result::SUCCESS;
        }

        state.sakura.themeKey = msg.themeKey;
        state.sakura.colorMap = Sakura::Themes.at(state.sakura.themeKey);
        state.sakura.runtime.update({
            .palette = &state.sakura.colorMap,
            .render = state.system.render.get(),
        });
        RethemeSurfaces(state.flowgraph.surfaces, state.sakura.colorMap);
        if (state.system.render) {
            state.system.render->setDarkAppearance(state.sakura.dark());
        }

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.interface.themeKey = state.sakura.themeKey;
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailOpenModal& msg) {
        state.modal.content = msg.content;
        state.modal.flowgraph = msg.flowgraph;
        if (msg.content == ModalContent::Dependencies) {
            state.modal.dependencyRequest = state.runtime.dependencyRequest;
        } else {
            state.modal.dependencyRequest.reset();
        }
        if (msg.content != ModalContent::RenameBlock) {
            state.modal.renameBlockOldName.reset();
        }
        return Result::SUCCESS;
    }

    Result handle(const MailCloseModal&) {
        state.modal.content.reset();
        state.modal.flowgraph.reset();
        state.modal.renameBlockOldName.reset();
        state.modal.dependencyRequest.reset();
        return Result::SUCCESS;
    }

    Result handle(const MailNotify& msg) {
        callbacks.notify(msg.type, msg.durationMs, msg.message);
        return Result::SUCCESS;
    }

    Result handle(const MailNotifyResult& msg) {
        callbacks.notifyResult(msg.result, msg.message);
        return Result::SUCCESS;
    }

    Result handle(const MailOpenUrl& msg) {
        if (const auto slug = Gallery::Slug(msg.url)) {
            const auto& flowgraphs = state.registry.flowgraphs;
            for (const auto& key : {GalleryActions::PublishedPrefix + *slug, *slug}) {
                if (std::any_of(flowgraphs.begin(), flowgraphs.end(), [&](const auto& entry) {
                        return entry.key == key;
                    })) {
                    callbacks.enqueueMail(MailOpenFlowgraphExample{.key = key});
                    return Result::SUCCESS;
                }
            }
        }
        if (const auto link = Docs::Resolve(msg.url); link && Docs::Bundled().find(link->slug).has_value()) {
            callbacks.enqueueMail(MailOpenDocs{.page = link->slug, .anchor = link->anchor});
            return Result::SUCCESS;
        }
        const Result result = Platform::OpenUrl(msg.url);
        if (msg.notifyResult) {
            callbacks.notifyResult(result, "");
        }
        return Result::SUCCESS;
    }

    Result handle(const MailCopyText& msg) {
        callbacks.setClipboardText(msg.value);
        callbacks.notify(Sakura::ToastType::Info, 3000, jst::fmt::format("{} copied to clipboard.", msg.label));
        return Result::SUCCESS;
    }

    Result handle(const MailQuit&) {
        return state.system.instance->requestStop();
    }

 private:
    void focusShown(WorkspaceTab tab) {
        if (state.tabShown(tab)) {
            Focus(state, tab);
        }
    }

    std::unordered_set<std::string> knownFlowgraphs;
    bool startTabWasShown = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_WORKBENCH_HH
