#include "state.hh"

namespace Jetstream {

bool DefaultCompositorState::SakuraState::dark() const {
    return themeKey != "Light";
}

std::string DefaultCompositorState::FlowgraphState::FieldKey(const std::string& flowgraphId,
                                                             const std::string& blockName,
                                                             const std::string& fieldName) {
    return jst::fmt::format("{}:{}:{}", flowgraphId, blockName, fieldName);
}

bool DefaultCompositorState::startTabShown() const {
    return interface.startTab || workspaceTabs() == 0;
}

bool DefaultCompositorState::tabShown(WorkspaceTab tab) const {
    switch (tab) {
        case WorkspaceTab::Start:
            return startTabShown();
        case WorkspaceTab::Settings:
            return interface.settingsTab;
        case WorkspaceTab::Docs:
            return interface.docsTab;
    }
    return false;
}

bool DefaultCompositorState::tabActive(WorkspaceTab tab) const {
    return interface.focusedTab == tab && tabShown(tab);
}

bool DefaultCompositorState::startTabActive() const {
    return tabActive(WorkspaceTab::Start);
}

bool DefaultCompositorState::settingsTabActive() const {
    return tabActive(WorkspaceTab::Settings);
}

bool DefaultCompositorState::docsTabActive() const {
    return tabActive(WorkspaceTab::Docs);
}

std::optional<std::string> DefaultCompositorState::activeFlowgraph() const {
    if (startTabActive() || settingsTabActive() || docsTabActive() || terminal.focused.has_value()) {
        return std::nullopt;
    }
    if (!interface.focusedFlowgraph.has_value() || !flowgraph.items.contains(*interface.focusedFlowgraph)) {
        return std::nullopt;
    }
    return interface.focusedFlowgraph;
}

bool DefaultCompositorState::flowgraphUnsaved(const std::string& id) const {
    const auto entry = flowgraph.history.find(id);
    return entry != flowgraph.history.end() && entry->second.unsaved();
}

bool DefaultCompositorState::flowgraphMayLoseChanges(const std::string& id) const {
    const auto entry = flowgraph.history.find(id);
    return entry != flowgraph.history.end() && (entry->second.unsaved() || !entry->second.settled());
}

bool DefaultCompositorState::tabsHidden() const {
    return workspaceTabs() == 0 && !interface.startTab && !interface.settingsTab && !interface.docsTab;
}

U64 DefaultCompositorState::workspaceTabs() const {
    return flowgraph.items.size() + terminal.tabs.size();
}

}  // namespace Jetstream
