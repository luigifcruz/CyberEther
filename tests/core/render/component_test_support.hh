#ifndef JETSTREAM_TEST_RENDER_COMPONENT_SUPPORT_HH
#define JETSTREAM_TEST_RENDER_COMPONENT_SUPPORT_HH

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include "jetstream/render/base/window.hh"

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
#include "jetstream/render/devices/metal/surface.hh"
#include "jetstream/render/devices/metal/texture.hh"
#include "jetstream/render/devices/metal/transfer.hh"
#endif

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
#include "jetstream/render/devices/vulkan/surface.hh"
#include "jetstream/render/devices/vulkan/texture.hh"
#include "jetstream/render/devices/vulkan/transfer.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#endif

namespace {

using namespace Jetstream;

template<DeviceType D>
class ComponentWindow final : public Render::Window {
 public:
    ComponentWindow() : Window(Config{}) {}
    std::vector<std::shared_ptr<Render::SurfaceImp<D>>> surfaces;
    Result collect(Render::Transfer::Batch& batch) { return collectTransfers(batch); }
    std::string info() const override { return "ComponentWindow"; }
    constexpr DeviceType device() const override { return D; }
    bool supportsEraseBlending() const override { return eraseSupported; }
    bool eraseSupported = true;

 protected:
    Result bindSurface(const std::shared_ptr<Render::Surface>& surface) override {
        surfaces.push_back(std::dynamic_pointer_cast<Render::SurfaceImp<D>>(surface));
        return surfaces.back() ? Result::SUCCESS : Result::ERROR;
    }
    Result unbindSurface(const std::shared_ptr<Render::Surface>& surface) override {
        std::erase(surfaces, surface);
        return Result::SUCCESS;
    }
    Result underlyingCreate() override { return Result::SUCCESS; }
    Result underlyingDestroy() override { return Result::SUCCESS; }
    Result underlyingBegin() override { return Result::SUCCESS; }
    Result underlyingEnd() override { return Result::SUCCESS; }
    Result underlyingSynchronize() override { return Result::SUCCESS; }

};

template<DeviceType D>
class ComponentTexture final : public Render::TextureImp<D> {
 public:
    using Render::TextureImp<D>::TextureImp;
    auto handle() const { return this->getHandle(); }
};

template<DeviceType D>
class ComponentSurface final : public Render::SurfaceImp<D> {
 public:
    using Render::SurfaceImp<D>::SurfaceImp;
    using Render::SurfaceImp<D>::prepare;
    Result encode(auto command, bool visible) {
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
        if constexpr (D == DeviceType::Metal) return this->draw(command, visible);
#endif
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
        if constexpr (D == DeviceType::Vulkan) return Render::SurfaceImp<D>::encode(command, visible);
#endif
    }
    void submitted() {
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
        if constexpr (D == DeviceType::Metal) this->commitDraw();
#endif
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
        if constexpr (D == DeviceType::Vulkan) this->commit();
#endif
    }
};

template<DeviceType D> class ComponentDriver;

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
template<> class ComponentDriver<DeviceType::Metal> {
    NS::SharedPtr<MTL::CommandQueue> queue = NS::TransferPtr(Backend::State<DeviceType::Metal>()->getDevice()->newCommandQueue());
    Render::TransferImp<DeviceType::Metal> transfer;

 public:
    ~ComponentDriver() { transfer.destroy(); }
    void render(ComponentWindow<DeviceType::Metal>& window, const Render::Surface* hidden = nullptr) {
        auto command = queue->commandBuffer();
        Render::Transfer::Batch batch;
        for (auto& surface : window.surfaces) REQUIRE(std::static_pointer_cast<ComponentSurface<DeviceType::Metal>>(surface)->prepare() == Result::SUCCESS);
        REQUIRE(window.collect(batch) == Result::SUCCESS);
        if (!batch.empty()) REQUIRE(transfer.encode(batch, command, 0) == Result::SUCCESS);
        for (auto& surface : window.surfaces) REQUIRE(std::static_pointer_cast<ComponentSurface<DeviceType::Metal>>(surface)->encode(command, surface.get() != hidden) == Result::SUCCESS);
        command->commit();
        command->waitUntilCompleted();
        REQUIRE(command->status() == MTL::CommandBufferStatusCompleted);
        for (auto& surface : window.surfaces) std::static_pointer_cast<ComponentSurface<DeviceType::Metal>>(surface)->submitted();
        batch.commit();
    }
    std::vector<F32> pixels(const std::shared_ptr<ComponentTexture<DeviceType::Metal>>& texture) {
        const auto size = texture->size();
        const U64 rowBytes = (size.x * 16 + 255) / 256 * 256;
        auto buffer = NS::TransferPtr(Backend::State<DeviceType::Metal>()->getDevice()->newBuffer(rowBytes * size.y, MTL::ResourceStorageModeShared));
        auto command = queue->commandBuffer();
        auto blit = command->blitCommandEncoder();
        blit->copyFromTexture(texture->handle(), 0, 0, MTL::Origin(0, 0, 0), MTL::Size(size.x, size.y, 1), buffer.get(), 0, rowBytes, rowBytes * size.y);
        blit->endEncoding();
        command->commit();
        command->waitUntilCompleted();
        REQUIRE(command->status() == MTL::CommandBufferStatusCompleted);
        std::vector<F32> out(size.x * size.y * 4);
        for (U64 y = 0; y < size.y; ++y) {
            const auto row = reinterpret_cast<const F32*>(static_cast<const U8*>(buffer->contents()) + rowBytes * y);
            std::copy_n(row, size.x * 4, out.data() + size.x * y * 4);
        }
        return out;
    }
};
#endif

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
template<> class ComponentDriver<DeviceType::Vulkan> {
    VkDevice device = Backend::State<DeviceType::Vulkan>()->getDevice();
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    Render::TransferImp<DeviceType::Vulkan> transfer;

    void begin() {
        REQUIRE(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
        VkCommandBufferBeginInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQUIRE(vkBeginCommandBuffer(command, &info) == VK_SUCCESS);
    }
    void submit() {
        REQUIRE(vkEndCommandBuffer(command) == VK_SUCCESS);
        VkSubmitInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &command;
        auto queue = Backend::State<DeviceType::Vulkan>()->getGraphicsQueue();
        REQUIRE(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS);
        REQUIRE(vkQueueWaitIdle(queue) == VK_SUCCESS);
    }

 public:
    ComponentDriver() {
        VkCommandPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = Backend::FindQueueFamilies(Backend::State<DeviceType::Vulkan>()->getPhysicalDevice()).graphicFamily.value();
        REQUIRE(vkCreateCommandPool(device, &info, nullptr, &pool) == VK_SUCCESS);
        VkCommandBufferAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        REQUIRE(vkAllocateCommandBuffers(device, &alloc, &command) == VK_SUCCESS);
        transfer.create(1);
    }
    ~ComponentDriver() {
        transfer.destroy();
        vkDestroyCommandPool(device, pool, nullptr);
    }
    void render(ComponentWindow<DeviceType::Vulkan>& window, const Render::Surface* hidden = nullptr) {
        for (auto& surface : window.surfaces) REQUIRE(std::static_pointer_cast<ComponentSurface<DeviceType::Vulkan>>(surface)->prepare() == Result::SUCCESS);
        begin();
        Render::Transfer::Batch batch;
        REQUIRE(window.collect(batch) == Result::SUCCESS);
        if (!batch.empty()) REQUIRE(transfer.encode(batch, command, 0) == Result::SUCCESS);
        for (auto& surface : window.surfaces) REQUIRE(std::static_pointer_cast<ComponentSurface<DeviceType::Vulkan>>(surface)->encode(command, surface.get() != hidden) == Result::SUCCESS);
        submit();
        transfer.commit(batch);
        for (auto& surface : window.surfaces) std::static_pointer_cast<ComponentSurface<DeviceType::Vulkan>>(surface)->submitted();
        batch.commit();
    }
    std::vector<F32> pixels(const std::shared_ptr<ComponentTexture<DeviceType::Vulkan>>& texture) {
        const auto size = texture->size();
        const U64 bytes = size.x * size.y * 16;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        REQUIRE(vkCreateBuffer(device, &info, nullptr, &buffer) == VK_SUCCESS);
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex = Backend::FindMemoryType(Backend::State<DeviceType::Vulkan>()->getPhysicalDevice(), requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        REQUIRE(vkAllocateMemory(device, &alloc, nullptr, &memory) == VK_SUCCESS);
        REQUIRE(vkBindBufferMemory(device, buffer, memory, 0) == VK_SUCCESS);
        begin();
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = texture->handle();
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {static_cast<U32>(size.x), static_cast<U32>(size.y), 1};
        vkCmdCopyImageToBuffer(command, texture->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
        std::swap(barrier.oldLayout, barrier.newLayout);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        submit();
        void* data = nullptr;
        REQUIRE(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &data) == VK_SUCCESS);
        const auto first = static_cast<F32*>(data);
        std::vector<F32> out(first, first + size.x * size.y * 4);
        vkUnmapMemory(device, memory);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
        return out;
    }
};
#endif

}  // namespace

#endif
