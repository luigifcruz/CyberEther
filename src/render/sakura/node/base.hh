#pragma once

#include "../base.hh"

namespace Jetstream::Sakura::Private {

int NodeEditorObjectId(const std::string& id);
std::string NodeEditorPinKey(const NodeEditorPinRef& pin);
int NodeEditorPinObjectId(const NodeEditorPinRef& pin);

std::vector<std::string>& NodeEditorNodeStack();
std::unordered_map<int, std::string>& NodeEditorNodeRegistry();
std::unordered_map<int, NodeEditorPinRef>& NodeEditorPinRegistry();
std::unordered_map<int, std::string>& NodeEditorLinkRegistry();
std::unordered_map<int, std::function<void(bool)>>& NodeEditorLinkHoverRegistry();
std::unordered_map<int, std::function<void(const Context&)>>& NodeEditorLinkTooltipRegistry();

void ClearNodeEditorRegistries();
void RegisterNodeEditorNode(const std::string& id);
void RegisterNodeEditorPin(const NodeEditorPinRef& pin);
void RegisterNodeEditorLink(const std::string& id);
void RegisterNodeEditorLinkHover(const std::string& id, std::function<void(bool)> onHover);
void RegisterNodeEditorLinkTooltip(const std::string& id, std::function<void(const Context&)> tooltip);

}  // namespace Jetstream::Sakura::Private
