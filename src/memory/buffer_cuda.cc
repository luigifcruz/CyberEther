#include "buffer_backend.hh"

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE

#include "jetstream/backend/base.hh"
#include "jetstream/backend/devices/cuda/helpers.hh"
#include "jetstream/logger.hh"
#include "jetstream/memory/macros.hh"
#include "jetstream/memory/devices/cuda/buffer.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "jetstream/memory/devices/vulkan/buffer.hh"
#endif

namespace Jetstream::detail {

namespace {

class CudaBackend final : public CudaBufferBackend, public Backend {
 public:
    CudaBackend() = default;
    ~CudaBackend() override {
        destroy();
    }

    DeviceType device() const override {
        return DeviceType::CUDA;
    }

    Result create(const U64& bytes, const Buffer::Config& config) override {
        destroy();

        sizeBytes = bytes;
        borrowed = false;
        ownsMemory = true;
        externalDevice = DeviceType::None;

        if (bytes == 0) {
            return Result::SUCCESS;
        }

        const auto& state = Jetstream::Backend::State<DeviceType::CUDA>();
        if (!state->isAvailable()) {
            JST_ERROR("[MEMORY:BUFFER:CUDA] CUDA is not available.");
            return Result::ERROR;
        }

        if (config.hostAccessible) {
            allocBytes = JST_PAGE_ALIGNED_SIZE(bytes);
            JST_CUDA_CHECK(cudaMallocManaged(&buffer, allocBytes), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to allocate managed memory: {}.", err);
            });
            allocationKind = AllocationKind::Managed;
            hostAccessibleFlag = true;
            locationState = Location::Unified;
#if !defined(JST_OS_WINDOWS)
        } else if (state->canExportDeviceMemory()) {
            CUmemAllocationProp allocationProp = {};
            allocationProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            allocationProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            allocationProp.location.id = state->getDeviceId();
            allocationProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

            U64 granularity = 0;
            JST_CUDA_CHECK(cuMemGetAllocationGranularity(&granularity,
                                                         &allocationProp,
                                                         CU_MEM_ALLOC_GRANULARITY_MINIMUM), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to get allocation granularity: {}.", err);
            });

            allocBytes = JST_ROUND_UP(bytes, granularity);

            JST_CUDA_CHECK(cuMemCreate(&allocHandle, allocBytes, &allocationProp, 0), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to allocate device memory: {}.", err);
            });

            JST_CUDA_CHECK(cuMemAddressReserve(&devicePtr, allocBytes, 0, 0, 0), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to reserve virtual address space: {}.", err);
            });

            JST_CUDA_CHECK(cuMemMap(devicePtr, allocBytes, 0, allocHandle, 0), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to map allocated memory: {}.", err);
            });

            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = state->getDeviceId();
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            JST_CUDA_CHECK(cuMemSetAccess(devicePtr, allocBytes, &accessDesc, 1), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to set memory access flags: {}.", err);
            });

            buffer = reinterpret_cast<void*>(devicePtr);
            allocationKind = AllocationKind::VirtualMemory;
            hostAccessibleFlag = false;
            locationState = Location::Device;
#endif
        } else {
            allocBytes = JST_PAGE_ALIGNED_SIZE(bytes);
            JST_CUDA_CHECK(cudaMalloc(&buffer, allocBytes), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to allocate device memory: {}.", err);
            });
            allocationKind = AllocationKind::Device;
            hostAccessibleFlag = false;
            locationState = Location::Device;
        }

        JST_CUDA_CHECK(cudaMemset(buffer, 0, allocBytes), [&] {
            JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to clear memory: {}.", err);
        });

        return Result::SUCCESS;
    }

    Result create(void* pointer, const U64& bytes) override {
        (void)pointer;
        (void)bytes;
        JST_ERROR("[MEMORY:BUFFER:CUDA] Borrowed raw-pointer create is not supported.");
        return Result::ERROR;
    }

    Result create(const Backend& source) override {
        destroy();

        if (source.size() == 0) {
            sizeBytes = 0;
            return Result::SUCCESS;
        }

        const auto& state = Jetstream::Backend::State<DeviceType::CUDA>();
        if (!state->isAvailable()) {
            JST_ERROR("[MEMORY:BUFFER:CUDA] CUDA is not available.");
            return Result::ERROR;
        }

        if (source.device() == DeviceType::CPU) {
            if (!state->canImportHostMemory()) {
                JST_ERROR("[MEMORY:BUFFER:CUDA] CUDA cannot import host memory.");
                return Result::ERROR;
            }

            void* hostPtr = const_cast<void*>(source.rawHandle());
            if (!hostPtr) {
                JST_ERROR("[MEMORY:BUFFER:CUDA] CPU source buffer has null pointer.");
                return Result::ERROR;
            }

            if (!JST_IS_ALIGNED(hostPtr)) {
                JST_ERROR("[MEMORY:BUFFER:CUDA] CPU source pointer must be page aligned.");
                return Result::ERROR;
            }

            bool needsHostRegistration = true;
            cudaPointerAttributes attributes = {};
            const auto attrStatus = cudaPointerGetAttributes(&attributes, hostPtr);
            if (attrStatus == cudaSuccess) {
#if CUDART_VERSION >= 10000
                needsHostRegistration = (attributes.type == cudaMemoryTypeUnregistered);
#else
                needsHostRegistration = false;
#endif
            } else if (attrStatus == cudaErrorInvalidValue) {
                cudaGetLastError();
                needsHostRegistration = true;
            } else {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to query host pointer attributes: {}.", cudaGetErrorString(attrStatus));
                return Result::ERROR;
            }

            if (needsHostRegistration) {
                const auto registerSize = JST_PAGE_ALIGNED_SIZE(source.size());
                JST_CUDA_CHECK(cudaHostRegister(hostPtr, registerSize, cudaHostRegisterPortable), [&] {
                    JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to register host memory: {}.", err);
                });
                hostRegistrationOwned = true;
            }

            buffer = hostPtr;
            allocBytes = JST_PAGE_ALIGNED_SIZE(source.size());
            sizeBytes = source.size();
            borrowed = true;
            ownsMemory = false;
            hostAccessibleFlag = true;
            locationState = source.location();
            externalDevice = DeviceType::CPU;
            allocationKind = AllocationKind::ImportedHost;
            return Result::SUCCESS;
        }

#if defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE) && !defined(JST_OS_WINDOWS)
        if (source.device() == DeviceType::Vulkan) {
            if (!state->canImportDeviceMemory()) {
                JST_ERROR("[MEMORY:BUFFER:CUDA] CUDA cannot import device memory.");
                return Result::ERROR;
            }

            const auto* vulkanBackend = dynamic_cast<const VulkanBufferBackend*>(&source);
            if (!vulkanBackend) {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Vulkan source backend doesn't expose Vulkan interop.");
                return Result::ERROR;
            }

            const auto& vkState = Jetstream::Backend::State<DeviceType::Vulkan>();
            if (!vkState->canExportDeviceMemory()) {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Vulkan cannot export device memory.");
                return Result::ERROR;
            }

            auto& device = vkState->getDevice();

            VkMemoryGetFdInfoKHR fdInfo = {};
            fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
            fdInfo.memory = vulkanBackend->memory();
            fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

            auto vkGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
                vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
            if (!vkGetMemoryFdKHR) {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to load vkGetMemoryFdKHR.");
                return Result::ERROR;
            }

            JST_VK_CHECK(vkGetMemoryFdKHR(device, &fdInfo, &importedFd), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to export Vulkan memory handle.");
            });

            CUDA_EXTERNAL_MEMORY_HANDLE_DESC handleDesc = {};
            handleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
            handleDesc.handle.fd = importedFd;
            handleDesc.size = JST_PAGE_ALIGNED_SIZE(source.size());

            JST_CUDA_CHECK(cuImportExternalMemory(&externalMemory, &handleDesc), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to import Vulkan external memory: {}.", err);
            });

            CUDA_EXTERNAL_MEMORY_BUFFER_DESC mappedBufferDesc = {};
            mappedBufferDesc.offset = 0;
            mappedBufferDesc.size = JST_PAGE_ALIGNED_SIZE(source.size());

            CUdeviceptr importedPtr = 0;
            JST_CUDA_CHECK(cuExternalMemoryGetMappedBuffer(&importedPtr, externalMemory, &mappedBufferDesc), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] Failed to map imported memory into CUDA: {}.", err);
            });

            buffer = reinterpret_cast<void*>(importedPtr);
            allocBytes = mappedBufferDesc.size;
            sizeBytes = source.size();
            borrowed = true;
            ownsMemory = false;
            hostAccessibleFlag = (source.location() == Location::Host || source.location() == Location::Unified);
            locationState = source.location();
            externalDevice = DeviceType::Vulkan;
            allocationKind = AllocationKind::ImportedVulkan;
            return Result::SUCCESS;
        }
#endif

        JST_ERROR("[MEMORY:BUFFER:CUDA] Cannot mirror from device {}.", source.device());
        return Result::ERROR;
    }

    Result copyFrom(const Backend& source, void* context) override {
        JST_TRACE("[MEMORY:BUFFER:CUDA] Copying buffer.");

        if (sizeBytes == 0) {
            return Result::SUCCESS;
        }

        auto stream = static_cast<cudaStream_t>(context);

        if (stream) {
            JST_CUDA_CHECK(cudaMemcpyAsync(buffer,
                                           source.rawHandle(),
                                           source.size(),
                                           cudaMemcpyDefault,
                                           stream), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] cudaMemcpyAsync failed: {}.", err);
            });
        } else {
            JST_CUDA_CHECK(cudaMemcpy(buffer, source.rawHandle(), source.size(), cudaMemcpyDefault), [&] {
                JST_ERROR("[MEMORY:BUFFER:CUDA] cudaMemcpy failed: {}.", err);
            });
        }

        return Result::SUCCESS;
    }

    void* rawHandle() override {
        return buffer;
    }

    const void* rawHandle() const override {
        return buffer;
    }

    bool isBorrowed() const override {
        return borrowed;
    }

    Location location() const override {
        return locationState;
    }

    U64 size() const override {
        return sizeBytes;
    }

    bool hostAccessible() const override {
        return hostAccessibleFlag;
    }

    bool deviceNative() const override {
        return allocationKind == AllocationKind::Device ||
               allocationKind == AllocationKind::VirtualMemory ||
               allocationKind == AllocationKind::Managed;
    }

    bool exportableDeviceMemory() const override {
        return allocationKind == AllocationKind::VirtualMemory;
    }

    CUmemGenericAllocationHandle allocationHandle() const override {
        return allocHandle;
    }

    void destroy() override {
#if defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE) && !defined(JST_OS_WINDOWS)
        if (allocationKind == AllocationKind::ImportedVulkan) {
            if (externalMemory != nullptr) {
                cuDestroyExternalMemory(externalMemory);
                externalMemory = nullptr;
            }
            if (importedFd >= 0) {
                close(importedFd);
                importedFd = -1;
            }
        }
#endif

        if (hostRegistrationOwned && buffer) {
            auto status = cudaHostUnregister(buffer);
            if (status != cudaSuccess) {
                JST_WARN("[MEMORY:BUFFER:CUDA] Failed to unregister host memory: {}.", cudaGetErrorString(status));
            }
            hostRegistrationOwned = false;
        }

        if (ownsMemory && buffer) {
            switch (allocationKind) {
                case AllocationKind::VirtualMemory:
                    cuMemUnmap(devicePtr, allocBytes);
                    cuMemRelease(allocHandle);
                    cuMemAddressFree(devicePtr, allocBytes);
                    break;
                case AllocationKind::Managed:
                case AllocationKind::Device:
                    cudaFree(buffer);
                    break;
                default:
                    break;
            }
        }

        buffer = nullptr;
        devicePtr = 0;
        allocHandle = 0;
        sizeBytes = 0;
        allocBytes = 0;
        ownsMemory = true;
        borrowed = false;
        hostAccessibleFlag = false;
        locationState = Location::None;
        externalDevice = DeviceType::None;
        allocationKind = AllocationKind::None;
    }

 private:
    enum class AllocationKind {
        None,
        Device,
        Managed,
        VirtualMemory,
        ImportedHost,
        ImportedVulkan,
    };

    void* buffer = nullptr;
    CUdeviceptr devicePtr = 0;
    CUmemGenericAllocationHandle allocHandle = 0;

#if defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE) && !defined(JST_OS_WINDOWS)
    CUexternalMemory externalMemory = nullptr;
    int importedFd = -1;
#endif

    U64 sizeBytes = 0;
    U64 allocBytes = 0;
    bool ownsMemory = true;
    bool borrowed = false;
    bool hostAccessibleFlag = false;
    bool hostRegistrationOwned = false;
    Location locationState = Location::None;
    DeviceType externalDevice = DeviceType::None;
    AllocationKind allocationKind = AllocationKind::None;
};

}  // namespace

std::unique_ptr<Backend> CreateCudaBackend() {
    return std::make_unique<CudaBackend>();
}

}  // namespace Jetstream::detail

#endif  // JETSTREAM_BACKEND_CUDA_AVAILABLE
