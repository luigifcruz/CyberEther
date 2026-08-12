#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/window_attachment.hh"
#include "jetstream/render/components/font.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/platform.hh"

#include <algorithm>

namespace Jetstream::Render {

namespace {

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

    std::lock_guard<std::mutex> frameLock(newFrameQueueMutex);

    // Process pending attachment destruction.

    while (true) {
        std::shared_ptr<WindowAttachment> attachment;
        {
            std::lock_guard<std::mutex> lock(attachmentStateMutex);
            if (destroyQueue.empty()) {
                break;
            }
            attachment = destroyQueue.front().attachment;
            destroyQueue.pop();
            destroyingAttachments.insert(attachment.get());
        }

        const Result result = attachment->destroy();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            std::lock_guard<std::mutex> lock(attachmentStateMutex);
            destroyingAttachments.erase(attachment.get());
            destroyQueue.push({frameCount, attachment});
            return result;
        }
        {
            std::lock_guard<std::mutex> lock(attachmentStateMutex);
            destroyingAttachments.erase(attachment.get());
            const Result finalizeResult = finalizeAttachmentDestruction(attachment);
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
    }
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

    std::vector<std::shared_ptr<WindowAttachment>> expiredAttachments;

    {
        std::lock_guard<std::mutex> lock(attachmentStateMutex);

        int count = destroyQueue.size();
        while (count > 0) {
            auto [expiration, attachment] = destroyQueue.front();
            destroyQueue.pop();

            if (expiration <= frameCount) {
                expiredAttachments.push_back(attachment);
                destroyingAttachments.insert(attachment.get());
            } else {
                destroyQueue.push({expiration, attachment});
            }

            count--;
        }

        // Increment frame count.
        frameCount++;
    }

    Result result = res;
    for (auto& attachment : expiredAttachments) {
        const Result destroyResult = attachment->destroy();
        std::lock_guard<std::mutex> lock(attachmentStateMutex);
        destroyingAttachments.erase(attachment.get());
        if (destroyResult == Result::SUCCESS || destroyResult == Result::RELOAD) {
            const Result finalizeResult = finalizeAttachmentDestruction(attachment);
            if (finalizeResult != Result::SUCCESS &&
                finalizeResult != Result::RELOAD &&
                (result == Result::SUCCESS || result == Result::RELOAD)) {
                result = finalizeResult;
            }
        } else {
            destroyQueue.push({frameCount, attachment});
            if (result == Result::SUCCESS || result == Result::RELOAD) {
                result = destroyResult;
            }
        }
    }

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
    return resources;
}

Result Window::acquireSurfaceResources(const std::shared_ptr<Surface>& surface) {
    const auto resources = surfaceResourceList(surface);

    std::vector<std::shared_ptr<WindowAttachment>> acquired;
    acquired.reserve(resources.size());

    const auto rollback = [&]() {
        Result cleanupResult = Result::SUCCESS;
        for (auto it = acquired.rbegin(); it != acquired.rend(); ++it) {
            auto lease = resourceLeases.find(it->get());
            if (lease != resourceLeases.end() &&
                lease->second.framebufferOwner == surface.get()) {
                lease->second.framebufferOwner = nullptr;
            }
            const Result result = releaseSurfaceResource(*it);
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                cleanupResult = result;
            }
        }
        return cleanupResult;
    };

    for (const auto& resource : resources) {
        const bool framebuffer = resource.get() == surface->config.framebuffer.get();
        auto lease = resourceLeases.find(resource.get());
        if (lease == resourceLeases.end()) {
            Window* owner = resource->owner.load(std::memory_order_acquire);
            if (owner && owner != this) {
                const Result cleanupResult = rollback();
                return cleanupResult == Result::SUCCESS ? Result::ERROR : cleanupResult;
            }
            if (!owner) {
                Window* expected = nullptr;
                if (!resource->owner.compare_exchange_strong(expected,
                                                             this,
                                                             std::memory_order_acq_rel)) {
                    const Result cleanupResult = rollback();
                    return cleanupResult == Result::SUCCESS ? Result::ERROR : cleanupResult;
                }
            }

            const Result createResult = resource->create();
            if (createResult != Result::SUCCESS && createResult != Result::RELOAD) {
                resource->owner.store(nullptr, std::memory_order_release);
                const Result cleanupResult = rollback();
                return cleanupResult == Result::SUCCESS ? createResult : cleanupResult;
            }

            lease = resourceLeases.emplace(resource.get(), ResourceLease{}).first;
        } else if (lease->second.surfaceReferences == 0 && !lease->second.root) {
            const Result cleanupResult = rollback();
            return cleanupResult == Result::SUCCESS ? Result::ERROR : cleanupResult;
        }

        // Resizing recreates backend texture handles, so a framebuffer cannot
        // safely be cached by another Surface's program or render pass.
        if ((framebuffer && lease->second.surfaceReferences != 0) ||
            (!framebuffer && lease->second.framebufferOwner)) {
            JST_ERROR("[WINDOW] Surface framebuffers cannot be shared between surfaces.");
            const Result cleanupResult = rollback();
            return cleanupResult == Result::SUCCESS ? Result::ERROR : cleanupResult;
        }
        if (framebuffer) {
            lease->second.framebufferOwner = surface.get();
        }

        lease->second.surfaceReferences++;
        acquired.push_back(resource);
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
        const Result result = releaseSurfaceResource(resource);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            cleanupResult = result;
        }
    }

    return cleanupResult;
}

Result Window::releaseSurfaceResource(const std::shared_ptr<WindowAttachment>& resource) {
    auto lease = resourceLeases.find(resource.get());
    if (lease == resourceLeases.end() || lease->second.surfaceReferences == 0) {
        return Result::ERROR;
    }

    lease->second.surfaceReferences--;
    if (lease->second.surfaceReferences != 0 || lease->second.root) {
        return Result::SUCCESS;
    }
    if (lease->second.rootReleaseExpiration > frameCount) {
        destroyQueue.push({lease->second.rootReleaseExpiration, resource});
        return Result::SUCCESS;
    }

    const Result result = resource->destroy();
    if (result == Result::SUCCESS || result == Result::RELOAD) {
        resource->owner.store(nullptr, std::memory_order_release);
        resourceLeases.erase(lease);
    } else {
        destroyQueue.push({frameCount, resource});
    }
    return result;
}

Result Window::finalizeAttachmentDestruction(const std::shared_ptr<WindowAttachment>& attachment) {
    attachment->owner.store(nullptr, std::memory_order_release);

    if (attachment->type() == WindowAttachment::Type::Surface) {
        return releaseSurfaceResources(std::static_pointer_cast<Surface>(attachment));
    }

    if (attachment->type() == WindowAttachment::Type::Buffer ||
        attachment->type() == WindowAttachment::Type::Texture) {
        auto lease = resourceLeases.find(attachment.get());
        if (lease != resourceLeases.end()) {
            resourceLeases.erase(lease);
        }
    }

    return Result::SUCCESS;
}

Result Window::processAttachmentQueues() {
    std::lock_guard<std::mutex> lock(newFrameQueueMutex);
    std::lock_guard<std::mutex> stateLock(attachmentStateMutex);

    // Allocate space for belated attachments.

    std::vector<std::shared_ptr<WindowAttachment>> belated;
    belated.reserve(unbindQueue.size() + bindQueue.size());

    const auto releaseBelated = [&](const size_t start) {
        Result cleanupResult = Result::SUCCESS;
        for (size_t i = start; i < belated.size(); ++i) {
            const Result result = belated[i]->destroy();
            if (result == Result::SUCCESS || result == Result::RELOAD) {
                const Result finalizeResult = finalizeAttachmentDestruction(belated[i]);
                if (finalizeResult != Result::SUCCESS &&
                    finalizeResult != Result::RELOAD) {
                    cleanupResult = finalizeResult;
                }
            } else {
                destroyQueue.push({frameCount, belated[i]});
                cleanupResult = result;
            }
        }
        return cleanupResult;
    };

    // Process unbind attachment queue.

    while (!unbindQueue.empty()) {
        auto attachment = unbindQueue.front();
        unbindQueue.pop();

        const bool resource = attachment->type() == WindowAttachment::Type::Buffer ||
                              attachment->type() == WindowAttachment::Type::Texture;
        if (resource) {
            auto lease = resourceLeases.find(attachment.get());
            if (lease != resourceLeases.end()) {
                lease->second.root = false;
                lease->second.rootReleaseExpiration = frameCount + 4;
            }
            attachments.erase(std::remove(attachments.begin(), attachments.end(), attachment),
                              attachments.end());
            if (lease == resourceLeases.end() || lease->second.surfaceReferences == 0) {
                belated.push_back(attachment);
            }
            continue;
        }

        // Unregister Surface.

        if (attachment->type() == WindowAttachment::Type::Surface) {
            JST_CHECK(unbindSurface(std::dynamic_pointer_cast<Surface>(attachment)));
        }

        attachments.erase(std::remove(attachments.begin(), attachments.end(), attachment),
                          attachments.end());

        // Belay attachment destruction schedule.

        belated.push_back(attachment);
    }

    for (auto& attachment : belated) {
        destroyQueue.push({
            .expiration = frameCount + 4,  // TODO: Replace with value from implementation.
            .attachment = attachment,
        });
    }
    belated.clear();

    // Process bind attachment queue.

    while (!bindQueue.empty()) {
        auto attachment = bindQueue.front();
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
                    const Result cleanupResult = releaseBelated(0);
                    return cleanupResult == Result::SUCCESS ? Result::ERROR : cleanupResult;
                }
            } else if (owner != this) {
                const Result cleanupResult = releaseBelated(0);
                return cleanupResult == Result::SUCCESS ? Result::ERROR : cleanupResult;
            }

            auto lease = resourceLeases.find(attachment.get());
            if (lease == resourceLeases.end()) {
                const Result createResult = attachment->create();
                if (createResult != Result::SUCCESS && createResult != Result::RELOAD) {
                    attachment->owner.store(nullptr, std::memory_order_release);
                    const Result cleanupResult = releaseBelated(0);
                    return cleanupResult == Result::SUCCESS ? createResult : cleanupResult;
                }
                lease = resourceLeases.emplace(attachment.get(), ResourceLease{}).first;
            }
            lease->second.root = true;
            lease->second.rootReleaseExpiration = 0;
            if (!Contains(attachments, attachment)) {
                attachments.push_back(attachment);
            }
            continue;
        }

        auto surface = std::dynamic_pointer_cast<Surface>(attachment);
        if (surface) {
            const Result acquireResult = acquireSurfaceResources(surface);
            if (acquireResult != Result::SUCCESS && acquireResult != Result::RELOAD) {
                attachment->owner.store(nullptr, std::memory_order_release);
                const Result cleanupResult = releaseBelated(0);
                return cleanupResult == Result::SUCCESS ? acquireResult : cleanupResult;
            }
        }

        // Create attachment.

        const Result createResult = attachment->create();
        if (createResult != Result::SUCCESS && createResult != Result::RELOAD) {
            if (surface) {
                releaseSurfaceResources(surface);
            }
            attachment->owner.store(nullptr, std::memory_order_release);
            const Result cleanupResult = releaseBelated(0);
            return cleanupResult == Result::SUCCESS ? createResult : cleanupResult;
        }

        // Belay attachment binding.

        if (attachment->type() == WindowAttachment::Type::Surface) {
            belated.push_back(attachment);
        }

        if (attachment->type() != WindowAttachment::Type::Surface &&
            !Contains(attachments, attachment)) {
            attachments.push_back(attachment);
        }
    }

    for (size_t i = 0; i < belated.size(); ++i) {
        auto& attachment = belated[i];
        const Result bindResult = bindSurface(std::dynamic_pointer_cast<Surface>(attachment));
        if (bindResult != Result::SUCCESS && bindResult != Result::RELOAD) {
            const Result cleanupResult = releaseBelated(i);
            return cleanupResult == Result::SUCCESS ? bindResult : cleanupResult;
        }
        if (!Contains(attachments, attachment)) {
            attachments.push_back(attachment);
        }
    }
    belated.clear();

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
