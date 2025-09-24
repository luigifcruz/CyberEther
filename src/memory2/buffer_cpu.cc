#include "buffer_backend.hh"

#include <cstdlib>
#include <cstring>

#include "jetstream/logger.hh"
#include "jetstream/memory2/macros.hh"

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/backend/devices/metal/base.hh"
#endif

#ifdef JST_OS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream::mem2::detail {

namespace {

class CpuBackend final : public Backend {
 public:
    CpuBackend() = default;
    ~CpuBackend() override {
        destroy();
    }

    Device device() const override {
        return Device::CPU;
    }

    Result create(const U64& bytes) override {
        destroy();

        size_bytes = bytes;
        borrowed = false;
        owns_memory = true;
        location_state = Location::Host;

        if (bytes == 0) {
            data_ptr = nullptr;
            return Result::SUCCESS;
        }

#ifdef JST_OS_WINDOWS
        data_ptr = VirtualAlloc(nullptr, JST_PAGE_ALIGNED_SIZE(bytes), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!data_ptr) {
            JST_ERROR("[MEM2:BUFFER:CPU] Failed to allocate {} bytes.", bytes);
            return Result::ERROR;
        }
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, JST_PAGESIZE(), JST_PAGE_ALIGNED_SIZE(bytes)) != 0) {
            JST_ERROR("[MEM2:BUFFER:CPU] Failed to allocate {} bytes.", bytes);
            return Result::ERROR;
        }
        data_ptr = ptr;
#endif
        std::memset(data_ptr, 0, bytes);
        return Result::SUCCESS;
    }

    Result create(const Backend& source) override {
        // TODO: Implement CUDA -> CPU.
        // TODO: Implement Vulkan -> CPU.

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        if (source.device() == Device::Metal) {
            JST_TRACE("[MEM2:BUFFER:CPU] Mirroring Metal buffer.");

            if (source.location() == Location::Unified) {
                const auto* metal_buffer = static_cast<const MTL::Buffer*>(source.raw_handle());
                data_ptr = const_cast<MTL::Buffer*>(metal_buffer)->contents();
                if (!data_ptr) {
                    JST_ERROR("[MEM2:BUFFER:CPU] Metal buffer has no CPU-accessible contents.");
                    return Result::ERROR;
                }

                size_bytes = source.size();
                borrowed = true;
                owns_memory = false;
                location_state = Location::Unified;

                return Result::SUCCESS;
            }
        }
#endif

        JST_ERROR("[MEM2:BUFFER:CPU] Cannot mirror from device {}.", source.device());
        return Result::ERROR;
    }

    Result copy_from(const Backend& source) override {
        JST_TRACE("[MEM2:BUFFER:CPU] Copying buffer.");
        std::memcpy(data_ptr, source.raw_handle(), source.size());
        return Result::SUCCESS;
    }

    void* raw_handle() override {
        return data_ptr;
    }

    const void* raw_handle() const override {
        return data_ptr;
    }

    bool is_borrowed() const override {
        return borrowed;
    }

    Location location() const override {
        return location_state;
    }

    U64 size() const override {
        return size_bytes;
    }

    void destroy() override {
#ifdef JST_OS_WINDOWS
        if (data_ptr && owns_memory) {
            VirtualFree(data_ptr, 0, MEM_RELEASE);
        }
#else
        if (data_ptr && owns_memory) {
            std::free(data_ptr);
        }
#endif
        data_ptr = nullptr;
        size_bytes = 0;
        owns_memory = true;
        borrowed = false;
        location_state = Location::Host;
    }

 private:
    void* data_ptr = nullptr;
    U64 size_bytes = 0;
    bool owns_memory = true;
    bool borrowed = false;
    Location location_state = Location::Host;
};

}  // namespace

std::unique_ptr<Backend> CreateCpuBackend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace Jetstream::mem2::detail
