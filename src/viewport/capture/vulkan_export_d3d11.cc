#include "vulkan_export.hh"

#include "jetstream/viewport/capture/vulkan.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "jetstream/logger.hh"

#include <cstring>
#include <vector>

#include <d3d11_1.h>
#include <dxgi1_2.h>
#ifdef ERROR
#undef ERROR
#endif

namespace Jetstream::Viewport {

namespace {

class D3D11Export : public FrameExport {
 public:
    ~D3D11Export() override {
        destroy();
    }

    Result create(const VkExtent2D& extent, U32 family, U32 count) override {
        this->extent = extent;
        this->family = family;

        JST_CHECK(createDevice());

        targets.resize(count);
        for (auto& target : targets) {
            JST_CHECK(createTarget(target));
        }

        return Result::SUCCESS;
    }

    void destroy() override {
        if (!targets.empty()) {
            auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();

            for (auto& target : targets) {
                if (target.image != VK_NULL_HANDLE) {
                    vkDestroyImage(device, target.image, nullptr);
                }
                if (target.memory != VK_NULL_HANDLE) {
                    vkFreeMemory(device, target.memory, nullptr);
                }
                if (target.frame.handle != 0) {
                    CloseHandle(reinterpret_cast<HANDLE>(target.frame.handle));
                }
                if (target.texture) {
                    target.texture->Release();
                }
            }
            targets.clear();
        }

        if (d3dDevice) {
            d3dDevice->Release();
            d3dDevice = nullptr;
        }
    }

    void record(VkCommandBuffer commandBuffer, VkImage source, U32 index) override {
        const Target& target = targets[index];

        VkImageMemoryBarrier acquire = {};
        acquire.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        acquire.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        acquire.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        acquire.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        acquire.dstQueueFamilyIndex = family;
        acquire.image = target.image;
        acquire.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        acquire.subresourceRange.levelCount = 1;
        acquire.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &acquire);

        VkImageCopy region = {};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.layerCount = 1;
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.layerCount = 1;
        region.extent = {extent.width, extent.height, 1};

        vkCmdCopyImage(commandBuffer, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        VkImageMemoryBarrier release = acquire;
        release.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        release.dstAccessMask = 0;
        release.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        release.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        release.srcQueueFamilyIndex = family;
        release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &release);
    }

    const FrameCapture::SharedFrame& frame(U32 index) const override {
        return targets[index].frame;
    }

 private:
    struct Target {
        ID3D11Texture2D* texture = nullptr;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        FrameCapture::SharedFrame frame;
    };

    Result createDevice() {
        const U64 adapter = FrameCaptureVulkan::Device().luid;

        IDXGIFactory1* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
            JST_ERROR("[CAPTURE] Failed to create the DXGI factory.");
            return Result::ERROR;
        }

        IDXGIAdapter1* match = nullptr;
        IDXGIAdapter1* candidate = nullptr;
        for (UINT index = 0; !match && factory->EnumAdapters1(index, &candidate) != DXGI_ERROR_NOT_FOUND; ++index) {
            DXGI_ADAPTER_DESC1 description = {};
            candidate->GetDesc1(&description);
            U64 luid = 0;
            std::memcpy(&luid, &description.AdapterLuid, sizeof(luid));
            if (luid == adapter) {
                match = candidate;
            } else {
                candidate->Release();
            }
        }
        factory->Release();

        if (!match) {
            JST_ERROR("[CAPTURE] Failed to find the DXGI adapter of the Vulkan device.");
            return Result::ERROR;
        }

        const HRESULT created = D3D11CreateDevice(match, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
                                                  D3D11_SDK_VERSION, &d3dDevice, nullptr, nullptr);
        match->Release();
        if (FAILED(created)) {
            JST_ERROR("[CAPTURE] Failed to create the D3D11 sharing device ({:#x}).", static_cast<U32>(created));
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    Result createTarget(Target& target) {
        auto& device = Backend::State<DeviceType::Vulkan>()->getDevice();
        auto& physicalDevice = Backend::State<DeviceType::Vulkan>()->getPhysicalDevice();

        D3D11_TEXTURE2D_DESC textureInfo = {};
        textureInfo.Width = extent.width;
        textureInfo.Height = extent.height;
        textureInfo.MipLevels = 1;
        textureInfo.ArraySize = 1;
        textureInfo.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        textureInfo.SampleDesc.Count = 1;
        textureInfo.Usage = D3D11_USAGE_DEFAULT;
        textureInfo.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        textureInfo.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

        if (FAILED(d3dDevice->CreateTexture2D(&textureInfo, nullptr, &target.texture))) {
            JST_ERROR("[CAPTURE] Failed to create the shared frame texture.");
            return Result::ERROR;
        }

        IDXGIResource1* resource = nullptr;
        if (FAILED(target.texture->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void**>(&resource)))) {
            JST_ERROR("[CAPTURE] Failed to access the shared frame texture.");
            return Result::ERROR;
        }

        HANDLE handle = nullptr;
        const HRESULT exported = resource->CreateSharedHandle(nullptr,
                                                              DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                                              nullptr,
                                                              &handle);
        resource->Release();
        if (FAILED(exported)) {
            JST_ERROR("[CAPTURE] Failed to export the shared frame texture ({:#x}).", static_cast<U32>(exported));
            return Result::ERROR;
        }

        target.frame = {
            .handle = reinterpret_cast<U64>(handle),
            .pitch = static_cast<U64>(extent.width) * 4,
            .size = 0,
        };

        VkExternalMemoryImageCreateInfo externalInfo = {};
        externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;

        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.pNext = &externalInfo;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
        imageInfo.extent = {extent.width, extent.height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        JST_VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &target.image), [&]{
            JST_ERROR("[CAPTURE] Failed to create the shared frame image.");
        });

        VkMemoryRequirements requirements;
        vkGetImageMemoryRequirements(device, target.image, &requirements);

        U32 memoryTypeBits = requirements.memoryTypeBits;
        auto getHandleProperties = reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandlePropertiesKHR"));
        if (getHandleProperties) {
            VkMemoryWin32HandlePropertiesKHR handleProperties = {};
            handleProperties.sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR;
            if (getHandleProperties(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT, handle,
                                    &handleProperties) == VK_SUCCESS) {
                memoryTypeBits &= handleProperties.memoryTypeBits;
            }
        }

        VkMemoryDedicatedAllocateInfo dedicatedInfo = {};
        dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicatedInfo.image = target.image;

        VkImportMemoryWin32HandleInfoKHR importInfo = {};
        importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importInfo.pNext = &dedicatedInfo;
        importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
        importInfo.handle = handle;

        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.pNext = &importInfo;
        allocateInfo.allocationSize = requirements.size;
        if (FindExportMemoryType(physicalDevice, memoryTypeBits, allocateInfo.memoryTypeIndex) != Result::SUCCESS) {
            JST_ERROR("[CAPTURE] No memory type can hold the shared frame.");
            return Result::ERROR;
        }

        JST_VK_CHECK(vkAllocateMemory(device, &allocateInfo, nullptr, &target.memory), [&]{
            JST_ERROR("[CAPTURE] Failed to import the shared frame texture.");
        });

        JST_VK_CHECK(vkBindImageMemory(device, target.image, target.memory, 0), [&]{
            JST_ERROR("[CAPTURE] Failed to bind the shared frame texture.");
        });

        return Result::SUCCESS;
    }

    VkExtent2D extent = {};
    U32 family = 0;
    ID3D11Device* d3dDevice = nullptr;
    std::vector<Target> targets;
};

}  // namespace

std::unique_ptr<FrameExport> CreateD3D11Export() {
    return std::make_unique<D3D11Export>();
}

}  // namespace Jetstream::Viewport
