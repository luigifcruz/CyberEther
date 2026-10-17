#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/window_attachment.hh"
#include "jetstream/render/components/font.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/platform.hh"
#include "jetstream/viewport/adapters/generic.hh"
#include "jetstream/render/tools/imgui_internal.h"

#include <algorithm>

namespace Jetstream::Render {

namespace {

constexpr uint32_t kRetiredComponentFrames = 4;

template<typename T>
bool Contains(const std::vector<std::shared_ptr<T>>& items, const std::shared_ptr<T>& item) {
    return std::find(items.begin(), items.end(), item) != items.end();
}

bool QueueContains(std::queue<std::shared_ptr<WindowAttachment>> queue,
                   const std::shared_ptr<WindowAttachment>& attachment) {
    while (!queue.empty()) {
        if (queue.front() == attachment) {
            return true;
        }
        queue.pop();
    }

    return false;
}

bool QueueRemove(std::queue<std::shared_ptr<WindowAttachment>>& queue,
                 const std::shared_ptr<WindowAttachment>& attachment) {
    bool removed = false;
    std::queue<std::shared_ptr<WindowAttachment>> remaining;

    while (!queue.empty()) {
        if (queue.front() == attachment) {
            removed = true;
        } else {
            remaining.push(queue.front());
        }
        queue.pop();
    }

    queue.swap(remaining);

    return removed;
}

}  // namespace

Result Window::create() {
    JST_ASSERT(!created, "Window is already created.");

    // Set variables.

    _scalingFactor = 1.0f;
    _previousScalingFactor = 0.0f;
    graphicalLoopThreadStarted = false;
    frameCount = 0;

    std::lock_guard<std::mutex> frameLock(newFrameQueueMutex);

    // Call underlying create.
    Result result;
    try {
        result = underlyingCreate();
    } catch (...) {
        try {
            underlyingDestroy();
        } catch (...) {
        }
        throw;
    }
    if (result == Result::SUCCESS || result == Result::RELOAD) {
        created = true;
    } else {
        try {
            underlyingDestroy();
        } catch (...) {
            throw;
        }
    }
    return result;
}

void Window::setScale(F32 scale) {
    config.scale = scale;
}

Result Window::destroy() {
    JST_ASSERT(created, "Window is not created.");
    JST_ASSERT(!frameActive, "Cannot destroy a window during an active frame.");

    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        if (destructionInProgress) {
            JST_ERROR("[WINDOW] Window destruction is already in progress.");
            return Result::ERROR;
        }
        destructionInProgress = true;
    }

    const auto resetDestructionState = [&]() {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        destructionInProgress = false;
    };

    Result result;
    try {
        result = destroyInternal();
    } catch (...) {
        resetDestructionState();
        throw;
    }
    resetDestructionState();
    return result;
}

Result Window::destroyInternal() {
    JST_CHECK(processAttachmentQueues());

    // Unbind remaining components.

    while (true) {
        std::shared_ptr<Components::Generic> component;
        {
            std::lock_guard<std::mutex> lock(attachmentStateMutex);
            if (components.empty()) {
                break;
            }
            component = components.front();
        }

        JST_CHECK(unbind(component));
    }
    JST_CHECK(processAttachmentQueues());

    // Destroy remaining attachments.

    while (true) {
        std::shared_ptr<WindowAttachment> attachment;
        {
            std::lock_guard<std::mutex> lock(attachmentStateMutex);
            if (attachments.empty()) {
                break;
            }
            attachment = attachments.front();
        }

        JST_CHECK(unbind(attachment));
    }
    JST_CHECK(processAttachmentQueues());

    decltype(retiredComponents) expiredComponents;
    std::unique_lock<std::mutex> frameLock(newFrameQueueMutex);
    JST_CHECK(underlyingSynchronize());

    // Process pending attachment destruction.

    while (true) {
        PendingDestruction* stored;
        PendingDestruction pending;
        {
            std::lock_guard<std::mutex> lock(attachmentStateMutex);
            if (destroyQueue.empty()) {
                break;
            }
            stored = &destroyQueue.front();
            pending = *stored;
            if (!pending.attachment) {
                destroyQueue.pop();
                continue;
            }
            destroyingAttachments.insert(pending.attachment.get());
        }

        const auto attachment = pending.attachment;
        const Result result = destroyAttachment(pending);
        {
            std::lock_guard<std::mutex> lock(attachmentStateMutex);
            *stored = pending;
            destroyingAttachments.erase(attachment.get());
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                stored->expiration = frameCount;
                return result;
            }
            const Result finalizeResult = finalizeAttachmentDestruction(attachment, pending.releaseResources);
            destroyQueue.pop();
            if (finalizeResult != Result::SUCCESS &&
                finalizeResult != Result::RELOAD) {
                return finalizeResult;
            }
        }
    }

    // Destroy the backend only after attachments no longer need it.
    const Result res = underlyingDestroy();

    if (res == Result::SUCCESS || res == Result::RELOAD) {
        created = false;
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        expiredComponents.swap(retiredComponents);
    }
    frameLock.unlock();
    return res;
}

Result Window::begin() {
    // Assert window is started.
    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        JST_ASSERT(graphicalLoopThreadStarted, "Window is not started.");

        // Save current thread ID.
        graphicalLoopThreadId = std::this_thread::get_id();
    }

    // Process attachment queues.
    JST_CHECK(processAttachmentQueues());

    std::unique_lock<std::mutex> frameLock(newFrameQueueMutex);

    Result res;
    try {
        frameBackground = {0.0f, 0.0f, 0.0f, 1.0f};
        frameBackgroundDrawList = nullptr;
        res = underlyingBegin();
    } catch (...) {
        abortImguiFrame();
        underlyingCancel();
        throw;
    }

    if (res == Result::SUCCESS) {
        frameActive = true;
        frameLock.release();
    } else {
        abortImguiFrame();
    }

    return res;
}

void Window::prepareImgui() {
    ImGui::Render();
    auto* background = frameBackgroundDrawList;
    frameBackgroundDrawList = nullptr;

    const auto* data = ImGui::GetDrawData();
    const ImDrawIdx rectangle[] = {0, 1, 2, 0, 2, 3};
    bool folded = false;
    if (background && data && data->CmdListsCount > 0 && data->CmdLists[0] == background &&
        !background->CmdBuffer.empty() && background->IdxBuffer.Size >= 6) {
        auto& command = background->CmdBuffer[0];
        if (!command.UserCallback && command.IdxOffset == 0 && command.VtxOffset == 0 &&
            command.ElemCount >= 6 &&
            std::equal(std::begin(rectangle), std::end(rectangle), background->IdxBuffer.begin())) {
            command.IdxOffset += 6;
            command.ElemCount -= 6;
            folded = true;
        }
    }
    if (background && !folded) {
        frameBackground = {0.0f, 0.0f, 0.0f, 1.0f};
    }

    frameSampledTextures.clear();
    frameCustomDraws = data == nullptr;
    if (data) {
        for (const auto* list : data->CmdLists) {
            for (const auto& command : list->CmdBuffer) {
                frameCustomDraws |= command.UserCallback != nullptr;
                if (command.ElemCount > 0) {
                    frameSampledTextures.insert(command.TexRef.GetTexID());
                }
            }
        }
    }
}

bool Window::textureVisibleForPresentation(const Texture& texture) const {
    return frameCustomDraws || texture.visibleForPresentation(ImGui::GetFrameCount()) ||
           frameSampledTextures.contains(static_cast<ImTextureID>(texture.raw()));
}

Result Window::start() {
    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        JST_ASSERT(!graphicalLoopThreadStarted, "Window is already started.");
        graphicalLoopThreadStarted = true;
    }

    return Result::SUCCESS;
}

Result Window::stop() {
    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        JST_ASSERT(graphicalLoopThreadStarted, "Window is not started.");
        graphicalLoopThreadStarted = false;
    }

    return Result::SUCCESS;
}

Result Window::end() {
    JST_ASSERT(frameActive, "Window frame is not active.");
    std::vector<std::shared_ptr<Components::Generic>> expiredComponents;
    std::unique_lock<std::mutex> frameLock(newFrameQueueMutex, std::adopt_lock);

    // Call frame end.
    Result res;
    try {
        res = underlyingEnd();
    } catch (...) {
        abortImguiFrame();
        frameActive = false;
        try {
            underlyingCancel();
        } catch (...) {
        }
        throw;
    }
    frameActive = false;
    abortImguiFrame();

    // Process attachment destruction queue.

    std::vector<std::pair<PendingDestruction*, PendingDestruction>> expiredAttachments;
    bool retirementReady = false;

    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);

        if (res == Result::SUCCESS) {
            for (auto& [remaining, component] : retiredComponents) {
                if (remaining > 0) {
                    --remaining;
                }
            }
            retirementReady = !retiredComponents.empty() && retiredComponents.front().first == 0;
        }

        int count = destroyQueue.size();
        expiredAttachments.reserve(count);
        while (count > 0) {
            auto pending = destroyQueue.front();
            if (pending.attachment) {
                if (pending.expiration <= frameCount) {
                    destroyingAttachments.insert(pending.attachment.get());
                }
                destroyQueue.push(pending);
                if (pending.expiration <= frameCount) {
                    expiredAttachments.push_back({&destroyQueue.back(), pending});
                }
            }
            destroyQueue.pop();

            count--;
        }

        // Increment frame count.
        frameCount++;
    }

    Result result = res;
    if (retirementReady) {
        const Result syncResult = underlyingSynchronize();
        if (syncResult == Result::SUCCESS || syncResult == Result::RELOAD) {
            std::lock_guard<std::mutex> lock(attachmentStateMutex);
            while (!retiredComponents.empty() && retiredComponents.front().first == 0) {
                expiredComponents.push_back(std::move(retiredComponents.front().second));
                retiredComponents.pop_front();
            }
        } else {
            result = syncResult;
        }
    }
    for (auto& [stored, pending] : expiredAttachments) {
        const auto& attachment = pending.attachment;
        const Result destroyResult = destroyAttachment(pending);
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        *stored = pending;
        destroyingAttachments.erase(attachment.get());
        if (destroyResult == Result::SUCCESS || destroyResult == Result::RELOAD) {
            const Result finalizeResult = finalizeAttachmentDestruction(attachment, pending.releaseResources);
            stored->attachment.reset();
            if (finalizeResult != Result::SUCCESS &&
                finalizeResult != Result::RELOAD &&
                (result == Result::SUCCESS || result == Result::RELOAD)) {
                result = finalizeResult;
            }
        } else {
            stored->expiration = frameCount;
            if (result == Result::SUCCESS || result == Result::RELOAD) {
                result = destroyResult;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        while (!destroyQueue.empty() && !destroyQueue.front().attachment) {
            destroyQueue.pop();
        }
    }
    frameLock.unlock();
    return result;
}

Result Window::cancel() {
    JST_ASSERT(frameActive, "Window frame is not active.");
    std::unique_lock<std::mutex> frameLock(newFrameQueueMutex, std::adopt_lock);
    frameActive = false;

    abortImguiFrame();
    return underlyingCancel();
}

Result Window::synchronize() {
    std::lock_guard<std::mutex> frameLock(newFrameQueueMutex);

    // Call frame synchronize.
    return underlyingSynchronize();
}

void Window::abortImguiFrame() {
    ImGuiContext* context = ImGui::GetCurrentContext();
    if (context && context->WithinFrameScope) {
        ImGui::EndFrame();
    }
}

Result Window::bind(const std::shared_ptr<WindowAttachment>& attachment) {
    if (!attachment) {
        return Result::ERROR;
    }
    // Add attachment to bind queue.
    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        if (destructionInProgress) {
            JST_ERROR("[WINDOW] Cannot bind an attachment during window destruction.");
            return Result::ERROR;
        }
        Window* owner = attachment->owner.load(std::memory_order_acquire);
        if (owner && owner != this) {
            JST_ERROR("[WINDOW] Render attachments can only belong to one window.");
            return Result::ERROR;
        }
        if (QueueRemove(unbindQueue, attachment)) {
            return Result::SUCCESS;
        }
        if (Contains(attachments, attachment) || QueueContains(bindQueue, attachment)) {
            return Result::SUCCESS;
        }
        if (destroyingAttachments.contains(attachment.get())) {
            JST_ERROR("[WINDOW] Cannot rebind an attachment being destroyed.");
            return Result::ERROR;
        }

        auto pendingDestruction = destroyQueue;
        while (!pendingDestruction.empty()) {
            if (pendingDestruction.front().attachment == attachment) {
                JST_ERROR("[WINDOW] Cannot rebind an attachment pending destruction.");
                return Result::ERROR;
            }
            pendingDestruction.pop();
        }

        const bool leasedResource =
            (attachment->type() == WindowAttachment::Type::Buffer ||
             attachment->type() == WindowAttachment::Type::Texture) &&
            resourceLeases.contains(attachment.get());
        if (leasedResource) {
            if (!owner) {
                Window* expected = nullptr;
                if (!attachment->owner.compare_exchange_strong(expected,
                                                               this,
                                                               std::memory_order_acq_rel)) {
                    JST_ERROR("[WINDOW] Render attachment ownership changed while binding.");
                    return Result::ERROR;
                }
            }

            resourceLeases.at(attachment.get()).root = true;
            resourceLeases.at(attachment.get()).rootReleaseExpiration = 0;
            if (!Contains(attachments, attachment)) {
                attachments.push_back(attachment);
            }
            return Result::SUCCESS;
        }
        if (owner == this && !leasedResource) {
            JST_ERROR("[WINDOW] Cannot rebind an inactive owned attachment.");
            return Result::ERROR;
        }

        if (!owner) {
            Window* expected = nullptr;
            if (!attachment->owner.compare_exchange_strong(expected,
                                                           this,
                                                           std::memory_order_acq_rel)) {
                JST_ERROR("[WINDOW] Render attachment ownership changed while binding.");
                return Result::ERROR;
            }
        }
        bindQueue.push(attachment);
    }

    // Wait queue to be processed.

    if (shouldDeferAttachmentQueueProcessing()) {
        while (!attachmentQueueEmpty()) {
            std::this_thread::yield();
        }
    } else {
        JST_CHECK(processAttachmentQueues());
    }

    return Result::SUCCESS;
}

Result Window::unbind(const std::shared_ptr<WindowAttachment>& attachment) {
    // Skip if attachment is not bound.
    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        if (QueueRemove(bindQueue, attachment)) {
            attachment->owner.store(nullptr, std::memory_order_release);
            return Result::SUCCESS;
        }

        const auto it = std::find(attachments.begin(), attachments.end(), attachment);
        if (it == attachments.end()) {
            if (!QueueContains(unbindQueue, attachment)) {
                JST_WARN("[WINDOW] Trying to unbind an attachment that is not bound.");
            }
            return Result::SUCCESS;
        }

        if (QueueContains(unbindQueue, attachment)) {
            return Result::SUCCESS;
        }

        // Add attachment to unbind queue.
        unbindQueue.push(attachment);
    }

    // Wait queue to be processed.

    if (shouldDeferAttachmentQueueProcessing()) {
        while (!attachmentQueueEmpty()) {
            std::this_thread::yield();
        }
    } else {
        JST_CHECK(processAttachmentQueues());
    }

    return Result::SUCCESS;
}

std::vector<std::shared_ptr<WindowAttachment>> Window::surfaceResourceList(
    const std::shared_ptr<Surface>& surface) const {
    std::vector<std::shared_ptr<WindowAttachment>> resources;
    resources.reserve(surface->dependencyBuffers.size() + surface->dependencyTextures.size());
    for (const auto& buffer : surface->dependencyBuffers) {
        resources.push_back(buffer);
    }
    for (const auto& texture : surface->dependencyTextures) {
        resources.push_back(texture);
    }
    for (const auto& texture : surface->borrowedTextures) {
        if (!surface->dependencyTextures.contains(texture)) {
            resources.push_back(texture);
        }
    }
    return resources;
}

Result Window::extend(const std::shared_ptr<Surface>& surface,
                      const std::vector<std::shared_ptr<Program>>& programs,
                      const std::vector<std::shared_ptr<Kernel>>& kernels) {
    if (!surface) {
        return Result::ERROR;
    }
    if (programs.empty() && kernels.empty()) {
        return Result::SUCCESS;
    }
    if (!surface->appendable() || shouldDeferAttachmentQueueProcessing()) {
        return Result::SKIP;
    }
    if (frameActive.load(std::memory_order_acquire)) {
        return Result::SKIP;
    }

    std::lock_guard<std::mutex> lock(newFrameQueueMutex);
    std::lock_guard<std::mutex> stateLock(attachmentStateMutex);

    if (!Contains(attachments, std::static_pointer_cast<WindowAttachment>(surface)) ||
        !bindQueue.empty() || !unbindQueue.empty()) {
        return Result::SKIP;
    }

    JST_CHECK(surface->validateAppend(programs, kernels));

    for (const auto& program : programs) {
        for (const auto& texture : program->getConfig().textures) {
            const bool owned = surface->dependencyTextures.contains(texture);
            const bool borrowed = surface->borrowedTextures.contains(texture);
            if ((program->getConfig().borrowedTextures && !owned && !borrowed) ||
                (!program->getConfig().borrowedTextures && borrowed && !owned)) {
                return Result::SKIP;
            }
        }
    }

    auto nextConfig = surface->config;
    nextConfig.programs.insert(nextConfig.programs.end(), programs.begin(), programs.end());
    nextConfig.kernels.insert(nextConfig.kernels.end(), kernels.begin(), kernels.end());
    auto nextBuffers = surface->dependencyBuffers;
    auto nextDraws = surface->dependencyDraws;
    auto nextTextures = surface->dependencyTextures;
    auto nextBorrowed = surface->borrowedTextures;
    Surface::track(programs, kernels, nextBuffers, nextDraws, nextTextures, nextBorrowed);

    const auto resources = surface->untracked(programs, kernels);
    const U64 firstClaim = surface->objectClaims.size();
    JST_CHECK(surface->claimObjects(programs, kernels));
    bool resourcesAcquired = false;
    const auto rollback = [&]() {
        surface->releaseObjects(firstClaim);
        Result result = Result::SUCCESS;
        if (std::exchange(resourcesAcquired, false)) {
            for (auto it = resources.rbegin(); it != resources.rend(); ++it) {
                try {
                    const Result cleanup = releaseSurfaceResource(*it);
                    if (cleanup != Result::SUCCESS && cleanup != Result::RELOAD) {
                        result = cleanup;
                    }
                } catch (...) {
                    result = Result::ERROR;
                }
            }
        }
        return result;
    };
    Result result;
    try {
        const Result acquired = acquireSurfaceResources(surface, resources);
        if (acquired != Result::SUCCESS && acquired != Result::RELOAD) {
            surface->releaseObjects(firstClaim);
            return acquired;
        }
        resourcesAcquired = true;
        result = surface->createAppended(programs, kernels);
    } catch (...) {
        rollback();
        throw;
    }
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        const Result cleanup = rollback();
        return cleanup == Result::SUCCESS ? result : cleanup;
    }
    surface->config = std::move(nextConfig);
    surface->dependencyBuffers.swap(nextBuffers);
    surface->dependencyDraws.swap(nextDraws);
    surface->dependencyTextures.swap(nextTextures);
    surface->borrowedTextures.swap(nextBorrowed);
    surface->invalidate();

    return Result::SUCCESS;
}

Result Window::acquireSurfaceResources(const std::shared_ptr<Surface>& surface) {
    return acquireSurfaceResources(surface, surfaceResourceList(surface));
}

Result Window::acquireSurfaceResources(const std::shared_ptr<Surface>& surface,
                                       const std::vector<std::shared_ptr<WindowAttachment>>& resources) {
    std::vector<std::shared_ptr<WindowAttachment>> acquired;
    acquired.reserve(resources.size());

    const auto isBorrowed = [&](const std::shared_ptr<WindowAttachment>& resource) {
        const auto texture = std::dynamic_pointer_cast<Texture>(resource);
        return texture && surface->borrowedTextures.contains(texture) &&
               !surface->dependencyTextures.contains(texture);
    };

    const auto rollback = [&]() {
        Result cleanupResult = Result::SUCCESS;
        while (!acquired.empty()) {
            auto resource = std::move(acquired.back());
            acquired.pop_back();
            auto lease = resourceLeases.find(resource.get());
            if (lease != resourceLeases.end() &&
                lease->second.framebufferOwner == surface.get()) {
                lease->second.framebufferOwner = nullptr;
            }
            try {
                const Result result = releaseSurfaceResource(resource, isBorrowed(resource));
                if (result != Result::SUCCESS && result != Result::RELOAD) {
                    cleanupResult = result;
                }
            } catch (...) {
                cleanupResult = Result::ERROR;
            }
        }
        return cleanupResult;
    };

    const auto acquire = [&](const std::shared_ptr<WindowAttachment>& resource) -> Result {
        const bool framebuffer = resource.get() == surface->config.framebuffer.get();
        auto lease = resourceLeases.find(resource.get());
        if (isBorrowed(resource)) {
            if (lease == resourceLeases.end() ||
                resource->owner.load(std::memory_order_acquire) != this ||
                (lease->second.surfaceReferences == 0 && !lease->second.root) ||
                !std::static_pointer_cast<Texture>(resource)->allocated()) {
                return Result::ERROR;
            }
            lease->second.borrowedReferences++;
            acquired.push_back(resource);
            return Result::SUCCESS;
        }
        const bool needsCreation = lease == resourceLeases.end();
        if (needsCreation) {
            Window* owner = resource->owner.load(std::memory_order_acquire);
            if (owner && owner != this) {
                return Result::ERROR;
            }
            if (!owner) {
                Window* expected = nullptr;
                if (!resource->owner.compare_exchange_strong(expected,
                                                             this,
                                                             std::memory_order_acq_rel)) {
                    return Result::ERROR;
                }
            }

            try {
                lease = resourceLeases.emplace(resource.get(), ResourceLease{}).first;
            } catch (...) {
                resource->owner.store(nullptr, std::memory_order_release);
                throw;
            }
        } else if (lease->second.surfaceReferences == 0 && !lease->second.root) {
            return Result::ERROR;
        }

        // Resizing recreates backend texture handles, so a framebuffer cannot
        // safely be cached by another Surface's program or render pass.
        if ((framebuffer && lease->second.surfaceReferences != 0) ||
            (!framebuffer && lease->second.framebufferOwner)) {
            JST_ERROR("[WINDOW] Surface framebuffers cannot be shared between surfaces.");
            return Result::ERROR;
        }
        if (framebuffer) {
            lease->second.framebufferOwner = surface.get();
        }

        lease->second.surfaceReferences++;
        acquired.push_back(resource);
        if (needsCreation) {
            const Result result = resource->create();
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                return result;
            }
        }
        return Result::SUCCESS;
    };

    try {
        for (const auto& resource : resources) {
            const Result result = acquire(resource);
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                const Result cleanup = rollback();
                return cleanup == Result::SUCCESS ? result : cleanup;
            }
        }
    } catch (...) {
        rollback();
        throw;
    }

    return Result::SUCCESS;
}

Result Window::releaseSurfaceResources(const std::shared_ptr<Surface>& surface) {
    Result cleanupResult = Result::SUCCESS;
    for (const auto& resource : surfaceResourceList(surface)) {
        auto lease = resourceLeases.find(resource.get());
        if (lease != resourceLeases.end() &&
            lease->second.framebufferOwner == surface.get()) {
            lease->second.framebufferOwner = nullptr;
        }
        const auto texture = std::dynamic_pointer_cast<Texture>(resource);
        const bool borrowed = texture && surface->borrowedTextures.contains(texture) &&
                              !surface->dependencyTextures.contains(texture);
        try {
            const Result result = releaseSurfaceResource(resource, borrowed);
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                cleanupResult = result;
            }
        } catch (...) {
            cleanupResult = Result::ERROR;
        }
    }

    return cleanupResult;
}

Result Window::releaseSurfaceResource(const std::shared_ptr<WindowAttachment>& resource,
                                      bool borrowed) {
    auto lease = resourceLeases.find(resource.get());
    if (lease == resourceLeases.end()) {
        return Result::ERROR;
    }

    auto& references = borrowed ? lease->second.borrowedReferences : lease->second.surfaceReferences;
    if (references == 0) {
        return Result::ERROR;
    }
    references--;
    if (lease->second.surfaceReferences != 0 || lease->second.borrowedReferences != 0 || lease->second.root) {
        return Result::SUCCESS;
    }
    if (lease->second.rootReleaseExpiration > frameCount) {
        destroyQueue.push({lease->second.rootReleaseExpiration, resource});
        return Result::SUCCESS;
    }

    Result result;
    try {
        result = resource->destroy();
    } catch (...) {
        destroyQueue.push({frameCount, resource});
        throw;
    }
    if (result == Result::SUCCESS || result == Result::RELOAD) {
        resource->owner.store(nullptr, std::memory_order_release);
        resourceLeases.erase(lease);
    } else {
        destroyQueue.push({frameCount, resource});
    }
    return result;
}

Result Window::finalizeAttachmentDestruction(const std::shared_ptr<WindowAttachment>& attachment,
                                             bool releaseResources) {
    if (attachment->type() == WindowAttachment::Type::Surface) {
        auto surface = std::static_pointer_cast<Surface>(attachment);
        const Result result = releaseResources ? releaseSurfaceResources(surface) : Result::SUCCESS;
        surface->releaseObjects();
        attachment->owner.store(nullptr, std::memory_order_release);
        return result;
    }

    attachment->owner.store(nullptr, std::memory_order_release);

    if (attachment->type() == WindowAttachment::Type::Buffer ||
        attachment->type() == WindowAttachment::Type::Texture) {
        auto lease = resourceLeases.find(attachment.get());
        if (lease != resourceLeases.end()) {
            resourceLeases.erase(lease);
        }
    }

    return Result::SUCCESS;
}

Result Window::destroyAttachment(PendingDestruction& pending) {
    try {
        if (pending.unbindBeforeDestroy) {
            JST_CHECK(unbindSurface(std::static_pointer_cast<Surface>(pending.attachment)));
            pending.unbindBeforeDestroy = false;
        }
        if (pending.destroyBackend) {
            JST_CHECK(pending.attachment->destroy());
            pending.destroyBackend = false;
        }
        return Result::SUCCESS;
    } catch (...) {
        return Result::ERROR;
    }
}

Result Window::processAttachmentQueues() {
    std::lock_guard<std::mutex> lock(newFrameQueueMutex);
    std::lock_guard<std::mutex> stateLock(attachmentStateMutex);

    size_t pendingCount = destroyQueue.size();
    while (pendingCount-- > 0) {
        auto& pending = destroyQueue.front();
        if (!pending.attachment) {
            destroyQueue.pop();
            continue;
        }
        if (!pending.unbindBeforeDestroy) {
            destroyQueue.push(pending);
            destroyQueue.pop();
            continue;
        }
        const Result result = destroyAttachment(pending);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return result;
        }
        const Result finalized = finalizeAttachmentDestruction(pending.attachment, pending.releaseResources);
        destroyQueue.pop();
        JST_CHECK(finalized);
    }

    // Process unbind attachment queue.

    while (!unbindQueue.empty()) {
        auto attachment = unbindQueue.front();

        const bool resource = attachment->type() == WindowAttachment::Type::Buffer ||
                              attachment->type() == WindowAttachment::Type::Texture;
        if (resource) {
            auto lease = resourceLeases.find(attachment.get());
            if (lease == resourceLeases.end() ||
                (lease->second.surfaceReferences == 0 && lease->second.borrowedReferences == 0)) {
                destroyQueue.push({
                    .expiration = frameCount + 4,  // TODO: Replace with value from implementation.
                    .attachment = attachment,
                });
            }
            if (lease != resourceLeases.end()) {
                lease->second.root = false;
                lease->second.rootReleaseExpiration = frameCount + 4;
            }
            attachments.erase(std::remove(attachments.begin(), attachments.end(), attachment),
                              attachments.end());
            unbindQueue.pop();
            continue;
        }

        // Belay attachment destruction schedule.

        destroyQueue.push({frameCount + 4, attachment});
        auto& pending = destroyQueue.back();

        // Unregister Surface.

        if (attachment->type() == WindowAttachment::Type::Surface) {
            Result result;
            try {
                result = unbindSurface(std::dynamic_pointer_cast<Surface>(attachment));
            } catch (...) {
                pending.attachment.reset();
                throw;
            }
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                pending.attachment.reset();
                return result;
            }
        }

        attachments.erase(std::remove(attachments.begin(), attachments.end(), attachment),
                          attachments.end());

        unbindQueue.pop();
    }

    // Process bind attachment queue.

    // Allocate space for belated attachments.

    std::vector<PendingDestruction*> staged;
    staged.reserve(bindQueue.size());
    PendingDestruction* rootTransaction = nullptr;
    std::optional<ResourceLease> previousRoot;
    const auto rollback = [&]() {
        if (previousRoot) {
            resourceLeases.at(rootTransaction->attachment.get()) = *previousRoot;
            rootTransaction->attachment.reset();
            previousRoot.reset();
        }
        Result cleanup = Result::SUCCESS;
        for (auto* pending : staged) {
            if (!pending->attachment) {
                continue;
            }
            try {
                const Result result = destroyAttachment(*pending);
                if (result != Result::SUCCESS && result != Result::RELOAD) {
                    cleanup = result;
                    continue;
                }
                const Result finalized = finalizeAttachmentDestruction(pending->attachment, pending->releaseResources);
                pending->attachment.reset();
                if (finalized != Result::SUCCESS && finalized != Result::RELOAD) {
                    cleanup = finalized;
                }
            } catch (...) {
                cleanup = Result::ERROR;
            }
        }
        return cleanup;
    };
    const auto drain = [&]() -> Result {
        while (!bindQueue.empty()) {
            auto attachment = bindQueue.front();
            destroyQueue.push({frameCount, attachment, false, false, false});
            auto* pending = &destroyQueue.back();
            staged.push_back(pending);
            bindQueue.pop();

            const bool resource = attachment->type() == WindowAttachment::Type::Buffer ||
                                  attachment->type() == WindowAttachment::Type::Texture;
            if (resource) {
                Window* owner = attachment->owner.load(std::memory_order_acquire);
                if (!owner) {
                    Window* expected = nullptr;
                    if (!attachment->owner.compare_exchange_strong(expected,
                                                                   this,
                                                                   std::memory_order_acq_rel)) {
                        pending->attachment.reset();
                        return Result::ERROR;
                    }
                } else if (owner != this) {
                    pending->attachment.reset();
                    return Result::ERROR;
                }

                auto lease = resourceLeases.find(attachment.get());
                if (lease == resourceLeases.end()) {
                    lease = resourceLeases.emplace(attachment.get(), ResourceLease{}).first;
                    pending->destroyBackend = true;
                    JST_CHECK(attachment->create());
                } else {
                    if (lease->second.surfaceReferences == 0 && lease->second.borrowedReferences == 0 &&
                        !lease->second.root) {
                        pending->attachment.reset();
                        return Result::ERROR;
                    }
                    rootTransaction = pending;
                    previousRoot = lease->second;
                }
                if (!Contains(attachments, attachment)) {
                    attachments.push_back(attachment);
                }
                lease->second.root = true;
                lease->second.rootReleaseExpiration = 0;
                pending->attachment.reset();
                rootTransaction = nullptr;
                previousRoot.reset();
                continue;
            }

            auto surface = std::dynamic_pointer_cast<Surface>(attachment);
            if (surface) {
                JST_CHECK(surface->validateAppend(surface->config.programs, surface->config.kernels));
                JST_CHECK(surface->claimObjects(surface->config.programs, surface->config.kernels));
                JST_CHECK(acquireSurfaceResources(surface));
                pending->releaseResources = true;
            }

            // Create attachment.

            pending->destroyBackend = true;
            JST_CHECK(attachment->create());

            // Belay attachment binding.

            if (!surface) {
                if (!Contains(attachments, attachment)) {
                    attachments.push_back(attachment);
                }
                pending->attachment.reset();
            }
        }

        for (auto* pending : staged) {
            if (!pending->attachment) {
                continue;
            }
            auto& attachment = pending->attachment;
            pending->unbindBeforeDestroy = true;
            JST_CHECK(bindSurface(std::static_pointer_cast<Surface>(attachment)));
            if (!Contains(attachments, attachment)) {
                attachments.push_back(attachment);
            }
            pending->attachment.reset();
        }
        return Result::SUCCESS;
    };
    Result result;
    try {
        result = drain();
    } catch (...) {
        rollback();
        throw;
    }
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        const Result cleanup = rollback();
        return cleanup == Result::SUCCESS ? result : cleanup;
    }
    while (!destroyQueue.empty() && !destroyQueue.front().attachment) {
        destroyQueue.pop();
    }

    return Result::SUCCESS;
}

Result Window::collectTransfers(Transfer::Batch& batch) const {
    std::vector<std::shared_ptr<WindowAttachment>> snapshot;
    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        snapshot = attachments;
    }

    for (const auto& attachment : snapshot) {
        if (attachment->type() == WindowAttachment::Type::Surface) {
            std::static_pointer_cast<Surface>(attachment)->prepareFrame();
        }
    }

    for (const auto& attachment : snapshot) {
        switch (attachment->type()) {
            case WindowAttachment::Type::Buffer:
                batch.collect(std::static_pointer_cast<Buffer>(attachment));
                break;
            case WindowAttachment::Type::Texture:
                batch.collect(std::static_pointer_cast<Texture>(attachment));
                break;
            case WindowAttachment::Type::Surface:
                std::static_pointer_cast<Surface>(attachment)->collectTransfers(batch);
                break;
            default:
                break;
        }
    }

    for (const auto& attachment : snapshot) {
        if (attachment->type() != WindowAttachment::Type::Surface) {
            continue;
        }

        const auto& surface = std::static_pointer_cast<Surface>(attachment);
        if (surface->retained() && surface->affectedBy(batch)) {
            surface->invalidate();
        }
    }

    return Result::SUCCESS;
}

TitlebarChrome Window::titlebarChrome() const {
    const auto* viewport = underlyingViewport();
    if (!viewport) {
        return {};
    }

    auto chrome = viewport->titlebarChrome();
    if (_scalingFactor > 0.0f) {
        chrome.leadingInset /= _scalingFactor;
        chrome.height /= _scalingFactor;
    }
    return chrome;
}

void Window::beginWindowDrag() {
    if (auto* viewport = underlyingViewport()) {
        viewport->beginWindowDrag();
    }
}

void Window::zoomWindow() {
    if (auto* viewport = underlyingViewport()) {
        viewport->zoomWindow();
    }
}

void Window::updateScalingFactor(const Viewport::Generic& viewport) {
    _scalingFactor = viewport.scale(config.scale);

    _previousScalingFactor = _scalingFactor;
}

Extent2D<F32> Window::framebufferScale() const {
    const auto scale = ImGui::GetIO().DisplayFramebufferScale;
    return {scale.x, scale.y};
}

Extent2D<U64> Window::framebufferSize(const Extent2D<F32>& displaySize) const {
    const auto scale = framebufferScale();
    return {static_cast<U64>(displaySize.x * scale.x),
            static_cast<U64>(displaySize.y * scale.y)};
}

Result Window::bind(const std::shared_ptr<Components::Generic>& component) {
    // Call create on the component.
    JST_CHECK(component->create(this));

    // Push component to the list.
    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        if (!Contains(components, component)) {
            components.push_back(component);
        }
    }

    return Result::SUCCESS;
}

Result Window::unbind(const std::shared_ptr<Components::Generic>& component) {
    // Skip if component is not bound.
    size_t position = 0;

    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        const auto it = std::find(components.begin(), components.end(), component);
        if (it == components.end()) {
            JST_WARN("[WINDOW] Trying to unbind a component that is not bound.");
            return Result::SUCCESS;
        }

        position = it - components.begin();
        components.erase(it);
    }

    // Call destroy on the component.
    const auto res = component->destroy(this);
    if (res != Result::SUCCESS && res != Result::RELOAD) {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        const auto it = components.begin() + std::min(position, components.size());
        components.insert(it, component);
        return res;
    }

    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        retiredComponents.emplace_back(kRetiredComponentFrames, component);
    }

    return Result::SUCCESS;
}

bool Window::shouldDeferAttachmentQueueProcessing() {
    std::lock_guard<std::mutex> lock(attachmentStateMutex);
    return graphicalLoopThreadStarted &&
           frameCount > 0 &&
           graphicalLoopThreadId != std::this_thread::get_id();
}

bool Window::attachmentQueueEmpty() const {
    std::lock_guard<std::mutex> lock(attachmentStateMutex);
    return bindQueue.empty() && unbindQueue.empty();
}

bool Window::hasFont(const std::string& name) const {
    return fonts.contains(name);
}

Result Window::addFont(const std::string& name, const std::shared_ptr<Components::Font>& font) {
    // Check if font exists.

    if (hasFont(name)) {
        JST_ERROR("[WINDOW] Font '{}' already exists.", name);
        return Result::ERROR;
    }

    // Bind to window.

    JST_CHECK(this->bind(font));

    // Add to list.

    fonts[name] = font;

    return Result::SUCCESS;
}

Result Window::removeFont(const std::string& name) {
    // Check if font exists.

    if (!hasFont(name)) {
        JST_ERROR("[WINDOW] Font '{}' does not exist.", name);
        return Result::ERROR;
    }

    // Unbind from window.

    const auto& font = fonts.at(name);
    JST_CHECK(this->unbind(font));

    // Remove from list.

    fonts.erase(name);

    return Result::SUCCESS;
}

const std::shared_ptr<Components::Font>& Window::font(const std::string& name) const {
    return fonts.at(name);
}

}  // namespace Jetstream::Render
