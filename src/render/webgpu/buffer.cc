#include "jetstream/render/webgpu/buffer.hh"

namespace Jetstream::Render {

using Implementation = BufferImp<Device::WebGPU>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating buffer.");

    auto device = Backend::State<Device::WebGPU>()->getDevice();
    const auto& byteSize = config.size * config.elementByteSize;

    bufferBindingLayout = {};
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
            bufferBindingLayout.type = wgpu::BufferBindingType::ReadOnlyStorage;
            break;
        case Target::UNIFORM:
            bufferUsageFlag |= wgpu::BufferUsage::Uniform;
            bufferBindingLayout.type = wgpu::BufferBindingType::Uniform;
            break;
        case Target::STORAGE_DYNAMIC:
        case Target::UNIFORM_DYNAMIC:
            JST_ERROR("[WebGPU] Buffer usage type not supported.")
            return Result::ERROR;
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
