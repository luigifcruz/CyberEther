#include <jetstream/render/sakura/components/dockspace_window.hh>

#include "../helpers.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Jetstream::Sakura {

namespace {

using DockItem = DockspaceWindow::DockItem;
using DockLayout = DockspaceWindow::DockLayout;

F32 ClampRatio(const F32 ratio) {
    return std::clamp(ratio, 0.05f, 0.95f);
}

bool SameF32(const F32 lhs, const F32 rhs) {
    return std::abs(lhs - rhs) <= 1.0e-4f;
}

bool SameExtent(const Extent2D<F32>& lhs, const Extent2D<F32>& rhs) {
    return std::abs(lhs.x - rhs.x) <= 0.5f && std::abs(lhs.y - rhs.y) <= 0.5f;
}

template<typename T>
void HashCombine(std::size_t& seed, const T& value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::size_t HashLayout(const DockLayout& layout) {
    std::size_t seed = 0;
    HashCombine(seed, layout.direction.has_value());
    if (layout.direction.has_value()) {
        HashCombine(seed, static_cast<int>(*layout.direction));
    }
    HashCombine(seed, layout.ratio.has_value());
    if (layout.ratio.has_value()) {
        HashCombine(seed, static_cast<int>(ClampRatio(*layout.ratio) * 10000.0f));
    }
    HashCombine(seed, layout.items.has_value());
    if (layout.items.has_value()) {
        HashCombine(seed, layout.items->size());
        for (const auto& item : *layout.items) {
            HashCombine(seed, item.key);
            HashCombine(seed, item.order);
        }
    }
    HashCombine(seed, layout.children.has_value());
    if (layout.children.has_value()) {
        HashCombine(seed, layout.children->size());
        for (const auto& child : *layout.children) {
            HashCombine(seed, HashLayout(child));
        }
    }
    return seed;
}

std::size_t HashLayout(const std::optional<DockLayout>& layout) {
    return layout.has_value() ? HashLayout(*layout) : 0;
}

bool SameItems(const std::optional<std::vector<DockItem>>& lhs,
               const std::optional<std::vector<DockItem>>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
        return false;
    }
    if (!lhs.has_value()) {
        return true;
    }
    if (lhs->size() != rhs->size()) {
        return false;
    }
    for (U64 i = 0; i < lhs->size(); ++i) {
        if ((*lhs)[i].key != (*rhs)[i].key || (*lhs)[i].order != (*rhs)[i].order) {
            return false;
        }
    }
    return true;
}

bool SameLayout(const DockLayout& lhs, const DockLayout& rhs);

bool SameChildren(const std::optional<std::vector<DockLayout>>& lhs,
                  const std::optional<std::vector<DockLayout>>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
        return false;
    }
    if (!lhs.has_value()) {
        return true;
    }
    if (lhs->size() != rhs->size()) {
        return false;
    }
    for (U64 i = 0; i < lhs->size(); ++i) {
        if (!SameLayout((*lhs)[i], (*rhs)[i])) {
            return false;
        }
    }
    return true;
}

bool SameLayout(const DockLayout& lhs, const DockLayout& rhs) {
    if (lhs.direction != rhs.direction) {
        return false;
    }
    if (lhs.ratio.has_value() != rhs.ratio.has_value()) {
        return false;
    }
    if (lhs.ratio.has_value() && !SameF32(ClampRatio(*lhs.ratio), ClampRatio(*rhs.ratio))) {
        return false;
    }
    return SameItems(lhs.items, rhs.items) && SameChildren(lhs.children, rhs.children);
}

bool SameLayout(const std::optional<DockLayout>& lhs, const std::optional<DockLayout>& rhs) {
    if (lhs.has_value() != rhs.has_value()) {
        return false;
    }
    return !lhs.has_value() || SameLayout(*lhs, *rhs);
}

ImGuiDir ToImGuiDir(const DockLayout::Direction direction) {
    switch (direction) {
        case DockLayout::Direction::Left:
            return ImGuiDir_Left;
        case DockLayout::Direction::Right:
            return ImGuiDir_Right;
        case DockLayout::Direction::Up:
            return ImGuiDir_Up;
        case DockLayout::Direction::Down:
            return ImGuiDir_Down;
    }
    return ImGuiDir_Left;
}

bool HasLayoutContent(const DockLayout& layout) {
    return (layout.items.has_value() && !layout.items->empty()) ||
           (layout.children.has_value() && !layout.children->empty());
}

void DockItems(const std::optional<std::vector<DockItem>>& items,
               const ImGuiID nodeId,
               const std::unordered_map<std::string, std::string>& keyToLabel) {
    if (!items.has_value()) {
        return;
    }

    std::vector<DockItem> sortedItems = *items;
    std::sort(sortedItems.begin(), sortedItems.end(), [](const DockItem& lhs, const DockItem& rhs) {
        if (lhs.order == rhs.order) {
            return lhs.key < rhs.key;
        }
        return lhs.order < rhs.order;
    });

    for (const auto& item : sortedItems) {
        const auto labelIt = keyToLabel.find(item.key);
        if (labelIt == keyToLabel.end()) {
            continue;
        }
        ImGui::DockBuilderDockWindow(labelIt->second.c_str(), nodeId);
    }
}

void RestoreLayout(const DockLayout& layout,
                   const ImGuiID nodeId,
                   const std::unordered_map<std::string, std::string>& keyToLabel) {
    if (layout.children.has_value() && layout.direction.has_value() && layout.children->size() >= 2) {
        ImGuiID atDirNode = 0;
        ImGuiID oppositeNode = 0;
        const F32 ratio = ClampRatio(layout.ratio.value_or(0.5f));
        ImGui::DockBuilderSplitNode(nodeId,
                                    ToImGuiDir(*layout.direction),
                                    ratio,
                                    &atDirNode,
                                    &oppositeNode);
        RestoreLayout(layout.children->at(0), atDirNode, keyToLabel);
        RestoreLayout(layout.children->at(1), oppositeNode, keyToLabel);
        return;
    }

    DockItems(layout.items, nodeId, keyToLabel);
    if (!layout.children.has_value()) {
        return;
    }
    for (const auto& child : *layout.children) {
        RestoreLayout(child, nodeId, keyToLabel);
    }
}

DockLayout::Direction CaptureDirection(const ImGuiDockNode& first,
                                       const ImGuiDockNode& second,
                                       const ImGuiAxis splitAxis) {
    if (splitAxis == ImGuiAxis_Y || std::abs(first.Pos.y - second.Pos.y) > std::abs(first.Pos.x - second.Pos.x)) {
        return first.Pos.y <= second.Pos.y ? DockLayout::Direction::Up : DockLayout::Direction::Down;
    }
    return first.Pos.x <= second.Pos.x ? DockLayout::Direction::Left : DockLayout::Direction::Right;
}

F32 CaptureRatio(const ImGuiDockNode& node,
                 const ImGuiDockNode& child,
                 const DockLayout::Direction direction) {
    switch (direction) {
        case DockLayout::Direction::Left:
        case DockLayout::Direction::Right:
            return node.Size.x > 0.0f ? ClampRatio(child.Size.x / node.Size.x) : 0.5f;
        case DockLayout::Direction::Up:
        case DockLayout::Direction::Down:
            return node.Size.y > 0.0f ? ClampRatio(child.Size.y / node.Size.y) : 0.5f;
    }
    return 0.5f;
}

void CaptureWindowItem(ImGuiWindow* window,
                       U64& order,
                       const std::unordered_map<std::string, std::string>& labelToKey,
                       std::vector<DockItem>& items) {
    if (!window || !window->Name) {
        return;
    }

    const auto keyIt = labelToKey.find(window->Name);
    if (keyIt == labelToKey.end()) {
        return;
    }

    items.push_back({.key = keyIt->second, .order = order++});
}

std::vector<DockItem> CaptureItems(const ImGuiDockNode& node,
                                   const std::unordered_map<std::string, std::string>& labelToKey) {
    std::vector<DockItem> items;
    U64 order = 0;

    if (node.TabBar) {
        for (const auto& tab : node.TabBar->Tabs) {
            CaptureWindowItem(tab.Window, order, labelToKey, items);
        }
        return items;
    }

    for (ImGuiWindow* window : node.Windows) {
        CaptureWindowItem(window, order, labelToKey, items);
    }
    return items;
}

std::optional<DockLayout> CaptureLayout(const ImGuiDockNode* node,
                                        const std::unordered_map<std::string, std::string>& labelToKey) {
    if (!node) {
        return std::nullopt;
    }

    DockLayout layout;
    const auto items = CaptureItems(*node, labelToKey);
    if (!items.empty()) {
        layout.items = items;
    }

    if (node->ChildNodes[0] && node->ChildNodes[1]) {
        const ImGuiDockNode* first = node->ChildNodes[0];
        const ImGuiDockNode* second = node->ChildNodes[1];
        const DockLayout::Direction direction = CaptureDirection(*first, *second, node->SplitAxis);
        const ImGuiDockNode* atDir = first;
        const ImGuiDockNode* opposite = second;

        std::vector<DockLayout> children;
        const auto atDirLayout = CaptureLayout(atDir, labelToKey);
        const auto oppositeLayout = CaptureLayout(opposite, labelToKey);
        if (atDirLayout.has_value()) {
            children.push_back(*atDirLayout);
        }
        if (oppositeLayout.has_value()) {
            children.push_back(*oppositeLayout);
        }

        if (children.size() == 2) {
            layout.direction = direction;
            layout.ratio = CaptureRatio(*node, *atDir, direction);
            layout.children = std::move(children);
        } else if (children.size() == 1 && !layout.items.has_value()) {
            return children.front();
        }
    }

    if (!HasLayoutContent(layout)) {
        return std::nullopt;
    }
    return layout;
}

void CollectItemKeys(const DockLayout& layout, std::unordered_set<std::string>& keys) {
    if (layout.items.has_value()) {
        for (const auto& item : *layout.items) {
            if (!item.key.empty()) {
                keys.insert(item.key);
            }
        }
    }

    if (!layout.children.has_value()) {
        return;
    }
    for (const auto& child : *layout.children) {
        CollectItemKeys(child, keys);
    }
}

bool CapturedAllRequestedItems(const DockLayout& requested,
                               const std::optional<DockLayout>& captured) {
    std::unordered_set<std::string> requestedKeys;
    CollectItemKeys(requested, requestedKeys);
    if (requestedKeys.empty()) {
        return true;
    }
    if (!captured.has_value()) {
        return false;
    }

    std::unordered_set<std::string> capturedKeys;
    CollectItemKeys(*captured, capturedKeys);
    for (const auto& key : requestedKeys) {
        if (!capturedKeys.contains(key)) {
            return false;
        }
    }
    return true;
}

U64 CountKnownWindows(const ImGuiDockNode* node,
                      const std::unordered_map<std::string, std::string>& labelToKey) {
    if (!node) {
        return 0;
    }

    U64 count = 0;
    if (node->TabBar) {
        for (const auto& tab : node->TabBar->Tabs) {
            if (tab.Window && tab.Window->Name && labelToKey.contains(tab.Window->Name)) {
                ++count;
            }
        }
    } else {
        for (ImGuiWindow* window : node->Windows) {
            if (window && window->Name && labelToKey.contains(window->Name)) {
                ++count;
            }
        }
    }

    return count + CountKnownWindows(node->ChildNodes[0], labelToKey) +
           CountKnownWindows(node->ChildNodes[1], labelToKey);
}

}  // namespace

struct DockspaceWindow::Impl {
    Config config;
    std::string windowLabel;
    ImGuiID dockspaceId = 0;
    bool open = false;
    std::optional<Extent2D<F32>> lastPosition;
    std::optional<Extent2D<F32>> lastSize;
    std::optional<DockLayout> lastCapturedLayout;
    std::optional<std::size_t> lastRestoreHash;
    bool parentDockPending = false;
};

DockspaceWindow::DockspaceWindow() {
    this->impl = std::make_unique<Impl>();
}

DockspaceWindow::~DockspaceWindow() = default;
DockspaceWindow::DockspaceWindow(DockspaceWindow&&) noexcept = default;
DockspaceWindow& DockspaceWindow::operator=(DockspaceWindow&&) noexcept = default;

bool DockspaceWindow::update(Config config) {
    const bool resetWindowState = this->impl->config.id != config.id;
    const bool resetParentDock = resetWindowState ||
                                 this->impl->config.dockIntoParent != config.dockIntoParent ||
                                 this->impl->config.parentDockId != config.parentDockId;
    if (resetWindowState) {
        this->impl->lastPosition.reset();
        this->impl->lastSize.reset();
        this->impl->lastCapturedLayout.reset();
        this->impl->lastRestoreHash.reset();
    }
    if (resetParentDock) {
        this->impl->parentDockPending = config.dockIntoParent && config.parentDockId.has_value() &&
                                        *config.parentDockId != 0;
    }
    const std::string title = config.title.empty() ? config.id : config.title;
    this->impl->windowLabel = title + "###" + config.id;
    this->impl->dockspaceId = ImHashStr((config.id + ":dockspace").c_str());
    if (!config.restoreLayout) {
        this->impl->lastRestoreHash.reset();
    }
    this->impl->config = std::move(config);
    return true;
}

void DockspaceWindow::render(const Context& ctx, Child emptyContent) {
    const auto& config = impl->config;
    if (config.id.empty()) {
        return;
    }

    if (config.parentDockId.has_value() && *config.parentDockId != 0 && config.dockIntoParent &&
        impl->parentDockPending) {
        ImGui::SetNextWindowDockID(static_cast<ImGuiID>(*config.parentDockId), ImGuiCond_Always);
    }
    ImGui::SetNextWindowPos(Private::ToImVec2(Scale(ctx, config.position)), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(Private::ToImVec2(Scale(ctx, config.size)), ImGuiCond_FirstUseEver);

    ImGuiWindowClass windowClass;
    windowClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoCloseButton;
    ImGui::SetNextWindowClass(&windowClass);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    bool nextOpen = true;
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;
    if (impl->parentDockPending) {
        windowFlags |= ImGuiWindowFlags_NoFocusOnAppearing;
    }
    const bool expanded = ImGui::Begin(impl->windowLabel.c_str(), &nextOpen, windowFlags);
    ImGui::PopStyleVar();
    if (!nextOpen) {
        ImGui::End();
        if (impl->open) {
            impl->open = false;
            if (config.onClose) {
                config.onClose();
            }
        }
        return;
    }

    impl->open = true;
    if (impl->parentDockPending && ImGui::IsWindowDocked()) {
        impl->parentDockPending = false;
    }

    std::unordered_map<std::string, std::string> keyToLabel;
    std::unordered_map<std::string, std::string> labelToKey;
    for (const auto& dockable : config.dockables) {
        if (dockable.key.empty() || dockable.label.empty()) {
            continue;
        }
        keyToLabel[dockable.key] = dockable.label;
        labelToKey[dockable.label] = dockable.key;
    }

    ImVec2 dockspaceSize = ImGui::GetContentRegionAvail();
    if (dockspaceSize.x <= 0.0f || dockspaceSize.y <= 0.0f) {
        dockspaceSize = Private::ToImVec2(Scale(ctx, config.size));
    }
    dockspaceSize.x = std::max(1.0f, dockspaceSize.x);
    dockspaceSize.y = std::max(1.0f, dockspaceSize.y);
    const bool shouldRestore = config.restoreLayout && config.layout.has_value() &&
                               (!impl->lastRestoreHash.has_value() ||
                                *impl->lastRestoreHash != HashLayout(config.layout));
    const std::size_t restoreHash = HashLayout(config.layout);
    bool restoredThisFrame = false;
    if (shouldRestore) {
        if (!ImGui::DockBuilderGetNode(impl->dockspaceId)) {
            ImGui::DockBuilderAddNode(impl->dockspaceId, ImGuiDockNodeFlags_DockSpace);
        }
        ImGui::DockBuilderSetNodeSize(impl->dockspaceId, dockspaceSize);
        ImGui::DockBuilderRemoveNodeChildNodes(impl->dockspaceId);
        RestoreLayout(*config.layout, impl->dockspaceId, keyToLabel);
        ImGui::DockBuilderFinish(impl->dockspaceId);
        restoredThisFrame = true;
    }

    if (!expanded) {
        ImGui::DockSpace(impl->dockspaceId,
                         dockspaceSize,
                         static_cast<ImGuiDockNodeFlags>(
                             static_cast<int>(ImGuiDockNodeFlags_KeepAliveOnly) |
                             static_cast<int>(ImGuiDockNodeFlags_NoWindowMenuButton) |
                             static_cast<int>(ImGuiDockNodeFlags_NoCloseButton)));
        const ImGuiDockNode* root = ImGui::DockBuilderGetNode(impl->dockspaceId);
        const auto capturedLayout = CaptureLayout(root, labelToKey);
        const bool restoreCaptureComplete = !config.restoreLayout || !config.layout.has_value() ||
                                            CapturedAllRequestedItems(*config.layout, capturedLayout);
        if (restoredThisFrame && restoreCaptureComplete) {
            impl->lastRestoreHash = restoreHash;
        }
        ImGui::End();
        return;
    }

    const Extent2D<F32> position = Unscale(ctx, Private::ToExtent2D(ImGui::GetWindowPos()));
    const Extent2D<F32> size = Unscale(ctx, Private::ToExtent2D(ImGui::GetWindowSize()));
    if (config.onGeometry &&
        (!impl->lastPosition.has_value() || !impl->lastSize.has_value() ||
         !SameExtent(*impl->lastPosition, position) || !SameExtent(*impl->lastSize, size))) {
        impl->lastPosition = position;
        impl->lastSize = size;
        config.onGeometry(position, size);
    }

    ImGui::DockSpace(impl->dockspaceId,
                     dockspaceSize,
                     ImGuiDockNodeFlags_NoWindowMenuButton |
                     ImGuiDockNodeFlags_NoCloseButton);

    const ImGuiDockNode* root = ImGui::DockBuilderGetNode(impl->dockspaceId);
    const U64 knownWindowCount = CountKnownWindows(root, labelToKey);
    const auto capturedLayout = CaptureLayout(root, labelToKey);
    const bool restoreCaptureComplete = !config.restoreLayout || !config.layout.has_value() ||
                                        CapturedAllRequestedItems(*config.layout, capturedLayout);
    if (restoredThisFrame && restoreCaptureComplete) {
        impl->lastRestoreHash = restoreHash;
    }

    if (!restoredThisFrame && restoreCaptureComplete && config.onLayout &&
        !SameLayout(impl->lastCapturedLayout, capturedLayout)) {
        impl->lastCapturedLayout = capturedLayout;
        config.onLayout(capturedLayout);
    }

    if (knownWindowCount == 0 && emptyContent) {
        emptyContent(ctx);
    }

    ImGui::End();
}

}  // namespace Jetstream::Sakura
