#include "workbench.hh"

namespace Jetstream {

WorkbenchActions::WorkbenchActions(DefaultCompositorState& state,
                                   DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

void WorkbenchActions::Focus(DefaultCompositorState& state, std::optional<WorkspaceTab> tab) {
    state.interface.focusedTab = tab;
    state.interface.tabNavigationPending = true;
    state.terminal.focused.reset();
}

void WorkbenchActions::Open(DefaultCompositorState& state, WorkspaceTab tab) {
    auto& ui = state.interface;
    switch (tab) {
        case WorkspaceTab::Start:
            if (!ui.startTab) {
                ui.startTab = true;
                ui.startTabBaseline = state.workspaceTabs();
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

void WorkbenchActions::Close(DefaultCompositorState& state, WorkspaceTab tab) {
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

void WorkbenchActions::OpenSettings(DefaultCompositorState& state, SettingsSection section) {
    SettingsActions::Select(state, section);
    Open(state, WorkspaceTab::Settings);
}

Result WorkbenchActions::handle(const MailOpenStartTab&) {
    Open(state, WorkspaceTab::Start);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailCloseStartTab&) {
    Close(state, WorkspaceTab::Start);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailSetWelcomeSection& msg) {
    state.interface.welcomeSection = msg.section;
    state.interface.welcomePage = 0;
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailSetWelcomeSidebar& msg) {
    state.interface.welcomeSidebarOpen = msg.open;
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailSetWelcomeQuery& msg) {
    state.interface.welcomeQuery = msg.query;
    state.interface.welcomePage = 0;
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailSetWelcomeCategory& msg) {
    state.interface.welcomeCategory = msg.category;
    state.interface.welcomePage = 0;
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailSetWelcomePage& msg) {
    state.interface.welcomePage = msg.page;
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailFocusStartTab&) {
    focusShown(WorkspaceTab::Start);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailOpenSettings& msg) {
    OpenSettings(state, msg.section);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailCloseSettings&) {
    Close(state, WorkspaceTab::Settings);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailFocusSettings&) {
    focusShown(WorkspaceTab::Settings);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailOpenDocs& msg) {
    auto& ui = state.interface;
    if (!msg.page.empty() || !ui.docsTab) {
        ui.docsPage = Docs::Bundled().find(msg.page).has_value() ? msg.page : std::string(Docs::Home);
        ui.docsAnchor = msg.anchor;
        ++ui.docsNavigations;
    }
    Open(state, WorkspaceTab::Docs);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailCloseDocs&) {
    Close(state, WorkspaceTab::Docs);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailSetDocsQuery& msg) {
    state.interface.docsQuery = msg.query;
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailFocusDocs&) {
    focusShown(WorkspaceTab::Docs);
    return Result::SUCCESS;
}

void WorkbenchActions::settlePendingFocus() {
    auto& ui = state.interface;
    if (!ui.pendingFocusedFlowgraph || !state.flowgraph.items.contains(*ui.pendingFocusedFlowgraph)) {
        return;
    }
    ui.focusedFlowgraph = ui.pendingFocusedFlowgraph;
    ui.pendingFocusedFlowgraph.reset();
    Focus(state, std::nullopt);
}

void WorkbenchActions::reconcileFlowgraphs() {
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
        ui.quitting = false;
    }
    if (!ui.focusedFlowgraph.has_value() && !ui.pendingFocusedFlowgraph.has_value() && items.size() == 1) {
        ui.focusedFlowgraph = items.begin()->first;
    }
}

void WorkbenchActions::Forget(DefaultCompositorState& state, const std::string& flowgraph) {
    state.flowgraph.closedDescriptions.erase(flowgraph);
    state.flowgraph.environment.erase(flowgraph);
    state.flowgraph.selections.erase(flowgraph);
    if (state.interface.pendingFocusedFlowgraph == flowgraph) {
        state.interface.pendingFocusedFlowgraph.reset();
    }
}

void WorkbenchActions::reconcileModal() {
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

void WorkbenchActions::settleStartTab() {
    auto& ui = state.interface;
    const bool navigated = ui.tabNavigationPending;
    const U64 count = state.workspaceTabs();
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

Result WorkbenchActions::handle(const MailApplyTheme& msg) {
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

Result WorkbenchActions::handle(const MailOpenModal& msg) {
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
    if (msg.content == ModalContent::Feedback) {
        if (state.feedback.blockReport) {
            state.feedback.text.clear();
            state.feedback.debugInfo.clear();
            state.feedback.blockReport = false;
        }
        ++state.feedback.session;
    }
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailCloseModal&) {
    if (state.modal.content == ModalContent::Feedback) {
        state.feedback.text.clear();
        state.feedback.debugInfo.clear();
        state.feedback.blockReport = false;
    }
    state.modal.content.reset();
    state.modal.flowgraph.reset();
    state.modal.renameBlockOldName.reset();
    state.modal.dependencyRequest.reset();
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailNotify& msg) {
    callbacks.notify(msg.type, msg.durationMs, msg.message);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailNotifyResult& msg) {
    callbacks.notifyResult(msg.result, msg.message);
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailOpenUrl& msg) {
    if (const auto slug = GalleryActions::Remote::Slug(msg.url); slug && state.registry.flowgraph(*slug)) {
        callbacks.enqueueMail(MailOpenFlowgraphExample{.key = *slug});
        return Result::SUCCESS;
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

Result WorkbenchActions::handle(const MailOpenLink& msg) {
    const auto plugin = Plugin::Gallery::Link(msg.url);
    const auto flowgraph = GalleryActions::Remote::Link(msg.url);
    if (!plugin.has_value() && !flowgraph.has_value()) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         "This link is not supported. Update CyberEther and try again.");
        return Result::SUCCESS;
    }

    const auto& owner = plugin.has_value() ? plugin->owner : flowgraph->owner;
    if (owner != Home::Gallery::Origin) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         jst::fmt::format("The gallery of '{}' is not available in this version of CyberEther.", owner));
        return Result::SUCCESS;
    }

    if (plugin.has_value()) {
        callbacks.enqueueMail(MailOpenPluginDetails{.id = plugin->id});
        return Result::SUCCESS;
    }

    auto& requests = state.registry.requests;
    if (std::ranges::none_of(requests, [&](const auto& request) { return request.key == flowgraph->slug; })) {
        requests.push_back({.key = flowgraph->slug});
    }

    return Result::SUCCESS;
}

void WorkbenchActions::settleLinks() {
    auto& registry = state.registry;
    std::erase_if(registry.requests, [&](auto& request) {
        if (registry.flowgraph(request.key) != nullptr) {
            callbacks.enqueueMail(MailOpenFlowgraphExample{.key = request.key});
            return true;
        }
        if (registry.fetch.loading) {
            return false;
        }
        if (!request.refreshed) {
            registry.fetch.next = {};
            request.refreshed = true;
            return false;
        }
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         jst::fmt::format("The flowgraph '{}' is not available in the gallery.", request.key));
        return true;
    });
}

Result WorkbenchActions::handle(const MailCopyText& msg) {
    callbacks.setClipboardText(msg.value);
    callbacks.notify(Sakura::ToastType::Info, 3000, jst::fmt::format("{} copied to clipboard.", msg.label));
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailRevealPath& msg) {
    std::error_code ec;
    const auto file = std::filesystem::absolute(Platform::PathFromUtf8(msg.path), ec);
    const auto folder = ec ? std::filesystem::path() : file.parent_path();
    if (folder.empty() || Platform::OpenFolder(Platform::PathToUtf8(folder)) != Result::SUCCESS) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Failed to open the folder.");
    }
    return Result::SUCCESS;
}

Result WorkbenchActions::handle(const MailQuit&) {
    return Quit(state);
}

Result WorkbenchActions::Quit(DefaultCompositorState& state) {
    auto& ui = state.interface;
    if (!ui.quitting) {
        ui.discardedFlowgraphs.clear();
    }

    const auto pending = [&](const std::string& flowgraph) {
        return state.flowgraph.items.contains(flowgraph) && state.flowgraphMayLoseChanges(flowgraph) &&
               !ui.discardedFlowgraphs.contains(flowgraph);
    };
    std::optional<std::string> next;
    if (ui.focusedFlowgraph.has_value() && pending(*ui.focusedFlowgraph)) {
        next = ui.focusedFlowgraph;
    } else if (const auto entry = std::find_if(state.flowgraph.items.begin(), state.flowgraph.items.end(),
                                               [&](const auto& item) { return pending(item.first); });
               entry != state.flowgraph.items.end()) {
        next = entry->first;
    }

    if (next.has_value()) {
        ui.quitting = true;
        ui.closingFlowgraph = std::move(next);
        return Result::SUCCESS;
    }
    ui.quitting = false;
    ui.closingFlowgraph.reset();
    return state.system.instance->requestStop();
}

void WorkbenchActions::focusShown(WorkspaceTab tab) {
    if (state.tabShown(tab)) {
        Focus(state, tab);
    }
}

}  // namespace Jetstream
