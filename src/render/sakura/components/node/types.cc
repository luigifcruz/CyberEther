#include "base.hh"

namespace Jetstream::Sakura::Private {

int NodeEditorObjectId(const std::string& id) {
    return static_cast<int>(std::hash<std::string>{}(id));
}

std::string NodeEditorPinKey(const NodeEditor::PinRef& pin) {
    return pin.nodeId + ":" + (pin.isInput ? "in:" : "out:") + pin.pinId;
}

int NodeEditorPinObjectId(const NodeEditor::PinRef& pin) {
    return NodeEditorObjectId(NodeEditorPinKey(pin));
}

std::vector<std::string>& NodeEditorNodeStack() {
    static std::vector<std::string> stack;
    return stack;
}

std::unordered_map<int, std::string>& NodeEditorNodeRegistry() {
    static std::unordered_map<int, std::string> registry;
    return registry;
}

std::unordered_map<int, NodeEditor::PinRef>& NodeEditorPinRegistry() {
    static std::unordered_map<int, NodeEditor::PinRef> registry;
    return registry;
}

std::unordered_map<int, std::string>& NodeEditorLinkRegistry() {
    static std::unordered_map<int, std::string> registry;
    return registry;
}

std::unordered_map<int, std::function<void(bool)>>& NodeEditorLinkHoverRegistry() {
    static std::unordered_map<int, std::function<void(bool)>> registry;
    return registry;
}

std::unordered_map<int, std::function<void(const Context&)>>& NodeEditorLinkTooltipRegistry() {
    static std::unordered_map<int, std::function<void(const Context&)>> registry;
    return registry;
}

void ClearNodeEditorRegistries() {
    NodeEditorNodeStack().clear();
    NodeEditorNodeRegistry().clear();
    NodeEditorPinRegistry().clear();
    NodeEditorLinkRegistry().clear();
    NodeEditorLinkHoverRegistry().clear();
    NodeEditorLinkTooltipRegistry().clear();
}

void RegisterNodeEditorNode(const std::string& id) {
    NodeEditorNodeRegistry()[NodeEditorObjectId(id)] = id;
}

void RegisterNodeEditorPin(const NodeEditor::PinRef& pin) {
    NodeEditorPinRegistry()[NodeEditorPinObjectId(pin)] = pin;
}

void RegisterNodeEditorLink(const std::string& id) {
    NodeEditorLinkRegistry()[NodeEditorObjectId(id)] = id;
}

void RegisterNodeEditorLinkHover(const std::string& id, std::function<void(bool)> onHover) {
    NodeEditorLinkHoverRegistry()[NodeEditorObjectId(id)] = std::move(onHover);
}

void RegisterNodeEditorLinkTooltip(const std::string& id, std::function<void(const Context&)> tooltip) {
    NodeEditorLinkTooltipRegistry()[NodeEditorObjectId(id)] = std::move(tooltip);
}

}  // namespace Jetstream::Sakura::Private
