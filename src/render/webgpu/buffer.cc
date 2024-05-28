#include "jetstream/render/webgpu/buffer.hh"

namespace Jetstream::Render {

using Implementation = BufferImp<Device::WebGPU>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating buffer.");

    auto device = Backend::State<Device::WebGPU>()->getDevice();
    const auto& byteSize = config.size * config.elementByteSize;

    wgpu::BufferUsage bufferUsageFlag{};
    bufferUsageFlag |= wgpu::BufferUsage::CopySrc;
    bufferUsageFlag |= wgpu::BufferUsage::CopyDst;

    if ((config.target & Target::VERTEX) == Target::VERTEX) {
        bufferUsageFlag |= wgpu::BufferUsage::Vertex;
    }

    if ((config.target & Target::VERTEX_INDICES) == Target::VERTEX_INDICES) {
        bufferUsageFlag |= wgpu::BufferUsage::Index;
    }

    if ((config.target & Target::STORAGE) == Target::STORAGE ||
        (config.target & Target::STORAGE_DYNAMIC) == Target::STORAGE_DYNAMIC) {
        bufferUsageFlag |= wgpu::BufferUsage::Storage;
    }

    if ((config.target & Target::UNIFORM) == Target::UNIFORM ||
        (config.target & Target::UNIFORM_DYNAMIC) == Target::UNIFORM_DYNAMIC) {
        bufferUsageFlag |= wgpu::BufferUsage::Uniform;
    }

    wgpu::BufferDescriptor bufferDescriptor{};
    bufferDescriptor.size = byteSize;
    bufferDescriptor.usage = bufferUsageFlag;

    buffer = device.CreateBuffer(&bufferDescriptor);

    if (config.buffer) {
        JST_CHECK(update());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying buffer.");

    buffer.Destroy();

    return Result::SUCCESS;
}

Result Implementation::update() {
    return update(0, config.size);
}

Result Implementation::update(const U64& offset, const U64& size) {
    if (size == 0) {
        return Result::SUCCESS;
    }

    auto& device = Backend::State<Device::WebGPU>()->getDevice();

    const auto& byteOffset = offset * config.elementByteSize;
    const auto& byteSize = size * config.elementByteSize;

    device.GetQueue().WriteBuffer(buffer, byteOffset, (uint8_t*)config.buffer+byteOffset, byteSize);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
