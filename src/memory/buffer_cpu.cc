#include "buffer_backend.hh"

#include <cstdlib>
#include <cstring>

#include "jetstream/logger.hh"
#include "jetstream/memory/macros.hh"

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/backend/devices/metal/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/backend/devices/vulkan/base.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#endif

#ifdef JST_OS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream::detail {

namespace {

class CpuBackend final : public Backend {
 public:
    CpuBackend() = default;
    ~CpuBackend() override {
        destroy();
    }

    DeviceType device() const override {
        return DeviceType::CPU;
    }

    Result create(const U64& bytes, const Buffer::Config&) override {
        destroy();

        sizeBytes = bytes;
        borrowed = false;
        ownsMemory = true;
        locationState = Location::Host;

        if (bytes == 0) {
            dataPtr = nullptr;
            return Result::SUCCESS;
        }

#ifdef JST_OS_WINDOWS
        dataPtr = VirtualAlloc(nullptr, JST_PAGE_ALIGNED_SIZE(bytes), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!dataPtr) {
            JST_ERROR("[MEMORY:BUFFER:CPU] Failed to allocate {} bytes.", bytes);
            return Result::ERROR;
        }
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, JST_PAGESIZE(), JST_PAGE_ALIGNED_SIZE(bytes)) != 0) {
            JST_ERROR("[MEMORY:BUFFER:CPU] Failed to allocate {} bytes.", bytes);
            return Result::ERROR;
        }
        dataPtr = ptr;
#endif
        std::memset(dataPtr, 0, bytes);
        return Result::SUCCESS;
    }

    Result create(void* pointer, const U64& bytes) override {
        destroy();

        if (bytes > 0 && pointer == nullptr) {
            JST_ERROR("[MEMORY:BUFFER:CPU] Cannot borrow null pointer with non-zero size.");
            return Result::ERROR;
        }

        dataPtr = pointer;
        sizeBytes = bytes;
        borrowed = true;
        ownsMemory = false;
        locationState = Location::Host;
        externalDevice = DeviceType::CPU;

        return Result::SUCCESS;
    }

    Result create(const Backend& source) override {
        // TODO: Implement CUDA -> CPU.

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        if (source.device() == DeviceType::Metal) {
            JST_TRACE("[MEMORY:BUFFER:CPU] Mirroring Metal buffer.");

            if (source.location() == Location::Unified) {
                const auto* metalBuffer = static_cast<const MTL::Buffer*>(source.rawHandle());
                dataPtr = const_cast<MTL::Buffer*>(metalBuffer)->contents();
                if (!dataPtr) {
                    JST_ERROR("[MEMORY:BUFFER:CPU] Metal buffer has no CPU-accessible contents.");
                    return Result::ERROR;
                }

                sizeBytes = source.size();
                borrowed = true;
                ownsMemory = false;
                locationState = Location::Unified;
                externalDevice = DeviceType::Metal;

                return Result::SUCCESS;
            }
        }
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        if (source.device() == DeviceType::Vulkan) {
            JST_TRACE("[MEMORY:BUFFER:CPU] Mirroring Vulkan buffer.");

            // The Vulkan backend already maps memory, so we can just use its pointer.
            dataPtr = const_cast<void*>(source.rawHandle());
            if (!dataPtr) {
                JST_ERROR("[MEMORY:BUFFER:CPU] Vulkan buffer has no CPU-accessible contents.");
                return Result::ERROR;
            }

            sizeBytes = source.size();
            borrowed = true;
            ownsMemory = false;
            locationState = source.location();
            externalDevice = DeviceType::Vulkan;

            return Result::SUCCESS;
        }
#endif

        JST_ERROR("[MEMORY:BUFFER:CPU] Cannot mirror from device {}.", source.device());
        return Result::ERROR;
    }

    Result copyFrom(const Backend& source) override {
        JST_TRACE("[MEMORY:BUFFER:CPU] Copying buffer.");
        std::memcpy(dataPtr, source.rawHandle(), source.size());
        return Result::SUCCESS;
    }

    void* rawHandle() override {
        return dataPtr;
    }

    const void* rawHandle() const override {
        return dataPtr;
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

    void destroy() override {
#ifdef JST_OS_WINDOWS
        if (dataPtr && ownsMemory) {
            VirtualFree(dataPtr, 0, MEM_RELEASE);
        }
#else
        if (dataPtr && ownsMemory) {
            std::free(dataPtr);
        }
#endif
        dataPtr = nullptr;
        sizeBytes = 0;
        ownsMemory = true;
        borrowed = false;
        locationState = Location::Host;
        externalDevice = DeviceType::None;
    }

 private:
    void* dataPtr = nullptr;
    U64 sizeBytes = 0;
    bool ownsMemory = true;
    bool borrowed = false;
    Location locationState = Location::Host;
    DeviceType externalDevice = DeviceType::None;
};

}  // namespace

std::unique_ptr<Backend> CreateCpuBackend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace Jetstream::detail
