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
    switch (config.target) {
        case Target::VERTEX:
            bufferUsageFlag |= wgpu::BufferUsage::Vertex;
            break;
        case Target::VERTEX_INDICES:
            bufferUsageFlag |= wgpu::BufferUsage::Index;
            break;
        case Target::STORAGE:
            bufferUsageFlag |= wgpu::BufferUsage::Storage;
            break;
        case Target::UNIFORM:
            bufferUsageFlag |= wgpu::BufferUsage::Uniform;
            break;
        case Target::STORAGE_DYNAMIC:
        case Target::UNIFORM_DYNAMIC:
            JST_FATAL("[WebGPU] Buffer usage type not supported.")
            return Result::ERROR;
            break;
    }

    wgpu::BufferDescriptor bufferDescriptor{};
    bufferDescriptor.size = byteSize;
    bufferDescriptor.usage = bufferUsageFlag;

    buffer = device.CreateBuffer(&bufferDescriptor);

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

    // TODO: Implement this.

    JST_ERROR("not implemented: buffer");

    // const auto& byteOffset = offset * config.elementByteSize;
    // const auto& byteSize = size * config.elementByteSize;

    // if (!config.enableZeroCopy) {
    //     uint8_t* ptr = static_cast<uint8_t*>(buffer->contents());
    //     memcpy(ptr + byteOffset, (uint8_t*)config.buffer + byteOffset, byteSize);
    // }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
