#ifndef JETSTREAM_RENDER_SAKURA_NODE_BASE_HH
#define JETSTREAM_RENDER_SAKURA_NODE_BASE_HH

#include "../../helpers.hh"

#include <jetstream/render/sakura/components/node/editor.hh>

namespace Jetstream::Sakura::Private {

int NodeEditorObjectId(const std::string& id);
std::string NodeEditorPinKey(const NodeEditor::PinRef& pin);
int NodeEditorPinObjectId(const NodeEditor::PinRef& pin);

std::vector<std::string>& NodeEditorNodeStack();
std::unordered_map<int, std::string>& NodeEditorNodeRegistry();
std::unordered_map<int, NodeEditor::PinRef>& NodeEditorPinRegistry();
std::unordered_map<int, std::string>& NodeEditorLinkRegistry();
std::unordered_map<int, std::function<void(bool)>>& NodeEditorLinkHoverRegistry();
std::unordered_map<int, std::function<void(const Context&)>>& NodeEditorLinkTooltipRegistry();

void ClearNodeEditorRegistries();
void RegisterNodeEditorNode(const std::string& id);
void RegisterNodeEditorPin(const NodeEditor::PinRef& pin);
void RegisterNodeEditorLink(const std::string& id);
void RegisterNodeEditorLinkHover(const std::string& id, std::function<void(bool)> onHover);
void RegisterNodeEditorLinkTooltip(const std::string& id, std::function<void(const Context&)> tooltip);

}  // namespace Jetstream::Sakura::Private

#endif  // JETSTREAM_RENDER_SAKURA_NODE_BASE_HH
