#include "instance_remote_encoder.hh"

#include "jetstream/logger.hh"

#include <dlfcn.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include <ffnvcodec/dynlink_cuda.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

namespace Jetstream {

struct Instance::Remote::Encoder::CudaInput : public Input {
    static constexpr unsigned int kDedicatedMemory = 0x1;

    ~CudaInput() override {
        if (driver.library && cudaContext()) {
            CUcontext previous = nullptr;
            driver.pushContext(cudaContext());
            for (const auto& [handle, mapping] : mappings) {
                (void)handle;
                driver.memFree(mapping.pointer);
                driver.destroyMemory(mapping.memory);
            }
            driver.popContext(&previous);
        }
        mappings.clear();
        av_buffer_unref(&frames);
        av_buffer_unref(&device);
    }

    Result create(Encoder& encoder) override {
        if (!driver.load()) {
            JST_ERROR("[REMOTE] Failed to load the CUDA driver.");
            return Result::ERROR;
        }

        int index = -1;
        JST_CHECK(findDevice(encoder.config.adapter, index));

        const std::string ordinal = std::to_string(index);
        if (const int error = av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, ordinal.c_str(), nullptr, 0);
            error < 0) {
            JST_ERROR("[REMOTE] Failed to create the CUDA encoder device.");
            return Result::ERROR;
        }

        frames = av_hwframe_ctx_alloc(device);
        if (!frames) {
            JST_ERROR("[REMOTE] Failed to allocate the CUDA encoder frames.");
            return Result::ERROR;
        }

        auto* framesContext = reinterpret_cast<AVHWFramesContext*>(frames->data);
        framesContext->format = AV_PIX_FMT_CUDA;
        framesContext->sw_format = AV_PIX_FMT_BGRA;
        framesContext->width = static_cast<int>(encoder.config.size.x);
        framesContext->height = static_cast<int>(encoder.config.size.y);

        if (av_hwframe_ctx_init(frames) < 0) {
            JST_ERROR("[REMOTE] Failed to initialize the CUDA encoder frames.");
            return Result::ERROR;
        }

        encoder.context->pix_fmt = AV_PIX_FMT_CUDA;
        encoder.context->hw_frames_ctx = av_buffer_ref(frames);
        if (!encoder.context->hw_frames_ctx) {
            JST_ERROR("[REMOTE] Failed to reference the CUDA encoder frames.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    Result wrap(Encoder& encoder, const SharedFrame& frame, AVFrame* output) override {
        CUdeviceptr pointer = 0;
        JST_CHECK(map(frame, pointer));

        output->format = AV_PIX_FMT_CUDA;
        output->width = encoder.context->width;
        output->height = encoder.context->height;
        output->data[0] = reinterpret_cast<U8*>(pointer);
        output->linesize[0] = static_cast<int>(frame.pitch);
        output->hw_frames_ctx = av_buffer_ref(frames);
        output->buf[0] = encoder.borrow(output->data[0], frame.size);
        if (!output->hw_frames_ctx || !output->buf[0]) {
            JST_ERROR("[REMOTE] Failed to wrap the CUDA encoder frame.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    bool borrowed() const override {
        return true;
    }

 private:
    class Driver {
     public:
        ~Driver() {
            if (library) {
                dlclose(library);
            }
        }

        bool load() {
            library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
            if (!library) {
                return false;
            }

            return symbol(init, "cuInit") &&
                   symbol(deviceGetCount, "cuDeviceGetCount") &&
                   symbol(deviceGet, "cuDeviceGet") &&
                   symbol(deviceGetUuid, "cuDeviceGetUuid") &&
                   symbol(pushContext, "cuCtxPushCurrent_v2") &&
                   symbol(popContext, "cuCtxPopCurrent_v2") &&
                   symbol(memFree, "cuMemFree_v2") &&
                   symbol(importMemory, "cuImportExternalMemory") &&
                   symbol(mappedBuffer, "cuExternalMemoryGetMappedBuffer") &&
                   symbol(destroyMemory, "cuDestroyExternalMemory");
        }

        void* library = nullptr;
        tcuInit* init = nullptr;
        tcuDeviceGetCount* deviceGetCount = nullptr;
        tcuDeviceGet* deviceGet = nullptr;
        tcuDeviceGetUuid* deviceGetUuid = nullptr;
        tcuCtxPushCurrent_v2* pushContext = nullptr;
        tcuCtxPopCurrent_v2* popContext = nullptr;
        tcuMemFree_v2* memFree = nullptr;
        tcuImportExternalMemory* importMemory = nullptr;
        tcuExternalMemoryGetMappedBuffer* mappedBuffer = nullptr;
        tcuDestroyExternalMemory* destroyMemory = nullptr;

     private:
        template<typename T>
        bool symbol(T*& function, const char* name) {
            function = reinterpret_cast<T*>(dlsym(library, name));
            return function != nullptr;
        }
    };

    struct Mapping {
        CUexternalMemory memory = nullptr;
        CUdeviceptr pointer = 0;
    };

    CUcontext cudaContext() const {
        if (!device) {
            return nullptr;
        }
        const auto* deviceContext = reinterpret_cast<const AVHWDeviceContext*>(device->data);
        return static_cast<const AVCUDADeviceContext*>(deviceContext->hwctx)->cuda_ctx;
    }

    Result findDevice(const SharedDevice& adapter, int& index) {
        int count = 0;
        if (driver.init(0) != CUDA_SUCCESS || driver.deviceGetCount(&count) != CUDA_SUCCESS) {
            JST_ERROR("[REMOTE] Failed to enumerate CUDA devices.");
            return Result::ERROR;
        }

        for (int ordinal = 0; ordinal < count; ++ordinal) {
            CUdevice candidate = 0;
            CUuuid uuid = {};
            if (driver.deviceGet(&candidate, ordinal) == CUDA_SUCCESS &&
                driver.deviceGetUuid(&uuid, candidate) == CUDA_SUCCESS &&
                std::memcmp(uuid.bytes, adapter.uuid.data(), adapter.uuid.size()) == 0) {
                index = ordinal;
                return Result::SUCCESS;
            }
        }

        JST_ERROR("[REMOTE] No CUDA device matches the Vulkan device.");
        return Result::ERROR;
    }

    Result map(const SharedFrame& frame, CUdeviceptr& pointer) {
        if (const auto it = mappings.find(frame.handle); it != mappings.end()) {
            pointer = it->second.pointer;
            return Result::SUCCESS;
        }

        const int fd = dup(static_cast<int>(frame.handle));
        if (fd < 0) {
            JST_ERROR("[REMOTE] Failed to duplicate the shared frame handle.");
            return Result::ERROR;
        }

        CUcontext previous = nullptr;
        driver.pushContext(cudaContext());

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC handleInfo = {};
        handleInfo.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        handleInfo.handle.fd = fd;
        handleInfo.size = frame.size;
        handleInfo.flags = kDedicatedMemory;

        Mapping mapping;
        if (driver.importMemory(&mapping.memory, &handleInfo) != CUDA_SUCCESS) {
            close(fd);
            driver.popContext(&previous);
            JST_ERROR("[REMOTE] Failed to import the shared frame into CUDA.");
            return Result::ERROR;
        }

        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferInfo = {};
        bufferInfo.size = frame.size;

        if (driver.mappedBuffer(&mapping.pointer, mapping.memory, &bufferInfo) != CUDA_SUCCESS) {
            driver.destroyMemory(mapping.memory);
            driver.popContext(&previous);
            JST_ERROR("[REMOTE] Failed to map the shared frame into CUDA.");
            return Result::ERROR;
        }

        driver.popContext(&previous);

        mappings[frame.handle] = mapping;
        pointer = mapping.pointer;
        return Result::SUCCESS;
    }

    Driver driver;
    AVBufferRef* device = nullptr;
    AVBufferRef* frames = nullptr;
    std::unordered_map<U64, Mapping> mappings;
};

std::unique_ptr<Instance::Remote::Encoder::Input> Instance::Remote::Encoder::CreateCudaInput() {
    return std::make_unique<CudaInput>();
}

}  // namespace Jetstream
