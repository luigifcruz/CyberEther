#include "jetstream/render/devices/webgpu/buffer.hh"

namespace Jetstream::Render {

using Implementation = BufferImp<DeviceType::WebGPU>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating buffer.");

    if (config.enableZeroCopy) {
        JST_ERROR("[WebGPU] Zero-copy render buffers are not supported.");
        return Result::ERROR;
    }

    if (config.buffer) {
        JST_CHECK(validateUpdate(0, config.size));
    }

    auto device = Backend::State<DeviceType::WebGPU>()->getDevice();
    const auto& byteSize = config.size * config.elementByteSize;

    WGPUBufferUsage bufferUsageFlag = 0;
    bufferUsageFlag |= WGPUBufferUsage_CopySrc;
    bufferUsageFlag |= WGPUBufferUsage_CopyDst;

    if ((config.target & Target::VERTEX) == Target::VERTEX) {
        bufferUsageFlag |= WGPUBufferUsage_Vertex;
    }

    if ((config.target & Target::VERTEX_INDICES) == Target::VERTEX_INDICES) {
        bufferUsageFlag |= WGPUBufferUsage_Index;
    }

    if ((config.target & Target::STORAGE) == Target::STORAGE ||
        (config.target & Target::STORAGE_DYNAMIC) == Target::STORAGE_DYNAMIC) {
        bufferUsageFlag |= WGPUBufferUsage_Storage;
    }

    if ((config.target & Target::UNIFORM) == Target::UNIFORM ||
        (config.target & Target::UNIFORM_DYNAMIC) == Target::UNIFORM_DYNAMIC) {
        bufferUsageFlag |= WGPUBufferUsage_Uniform;
    }

    if ((config.target & Target::INDIRECT) == Target::INDIRECT) {
        bufferUsageFlag |= WGPUBufferUsage_Indirect;
    }

    if ((config.target & Target::STORAGE_DYNAMIC) == Target::STORAGE_DYNAMIC ||
        (config.target & Target::UNIFORM_DYNAMIC) == Target::UNIFORM_DYNAMIC) {
        JST_ERROR("[WebGPU] Buffer usage type not supported.")
        return Result::ERROR;
    }

    WGPUBufferDescriptor bufferDescriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    bufferDescriptor.size = byteSize;
    bufferDescriptor.usage = bufferUsageFlag;

    buffer = wgpuDeviceCreateBuffer(device, &bufferDescriptor);

    if (config.buffer) {
        JST_CHECK(update());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying buffer.");

    if (buffer) {
        wgpuBufferDestroy(buffer);
        wgpuBufferRelease(buffer);
        buffer = nullptr;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
