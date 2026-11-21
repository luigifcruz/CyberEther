#ifndef JETSTREAM_RENDER_TRANSFER_IMPL_HH
#define JETSTREAM_RENDER_TRANSFER_IMPL_HH

#include "jetstream/render/base/transfer.hh"
#include "jetstream/render/base/window_attachment.hh"

#include <unordered_map>
#include <unordered_set>

namespace Jetstream::Render {

class Program;
class Surface;

struct TransferRegistry;

struct TransferResource {
    std::weak_ptr<WindowAttachment> attachment;
    std::weak_ptr<TransferRegistry> registry;
    std::atomic_bool active = true;
    std::atomic_bool queued = false;
};

struct TransferResourceSet {
    struct Consumer {
        std::weak_ptr<Surface> surface;
        std::weak_ptr<Program> program;
        bool borrowed = false;
    };

    struct Binding {
        std::shared_ptr<WindowAttachment> attachment;
        std::shared_ptr<TransferResource> resource;
        std::vector<Consumer> consumers;
    };

    std::unordered_map<const WindowAttachment*, Binding> bindings;
};

struct TransferRegistry {
    std::mutex mutex;
    std::vector<std::shared_ptr<TransferResource>> pending;
    std::atomic_bool rescan = false;

    void notify(const std::shared_ptr<TransferResource>& resource) noexcept {
        if (!resource->active.load(std::memory_order_acquire) ||
            resource->queued.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        try {
            std::lock_guard lock(mutex);
            pending.push_back(resource);
        } catch (...) {
            resource->queued.store(false, std::memory_order_release);
            rescan.store(true, std::memory_order_release);
        }
    }

    std::vector<std::shared_ptr<TransferResource>> take() {
        std::lock_guard lock(mutex);
        auto result = std::move(pending);
        pending.clear();
        for (const auto& resource : result) {
            resource->queued.store(false, std::memory_order_release);
        }
        return result;
    }
};

struct Transfer::Batch::State {
    std::unordered_set<const Buffer*> collectedBuffers;
    std::unordered_set<const Texture*> collectedTextures;
    std::unordered_set<const Buffer*> pendingBuffers;
    std::unordered_set<const Texture*> pendingTextures;
    std::vector<std::shared_ptr<const TransferResourceSet>> snapshots;

    bool visited(const WindowAttachment* resource) const {
        return std::any_of(snapshots.begin(), snapshots.end(), [&](const auto& snapshot) {
            return snapshot->bindings.contains(resource);
        });
    }

    void remember(const std::shared_ptr<const TransferResourceSet>& resources) {
        if (std::find(snapshots.begin(), snapshots.end(), resources) == snapshots.end()) {
            snapshots.push_back(resources);
        }
    }
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_TRANSFER_IMPL_HH
