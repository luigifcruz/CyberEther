#include "jetstream/render/devices/metal/texture.hh"

#include <cstring>

namespace Jetstream::Render {

using Implementation = TextureImp<DeviceType::Metal>;

Implementation::TextureImp(const Config& config) : Texture(config) {
}

Result Implementation::create() {
    JST_DEBUG("[METAL] Creating texture.");

    if (config.buffer) {
        JST_CHECK(validateFillRow(0, config.size.y));
    }

    pixelFormat = ConvertPixelFormat(config.pfmt, config.ptype);

    auto textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
            pixelFormat, config.size.x, config.size.y, false);
    JST_ASSERT(textureDesc, "Failed to create texture descriptor.");

    if (config.multisampled) {
        textureDesc->setSampleCount(Backend::State<DeviceType::Metal>()->getMultisampling());
        textureDesc->setTextureType(MTL::TextureType2DMultisample);
    }

    // TODO: Check if memoryless is an option here.

    textureDesc->setUsage(MTL::TextureUsagePixelFormatView |
                          MTL::TextureUsageRenderTarget |
                          MTL::TextureUsageShaderRead);
    textureDesc->setStorageMode(MTL::StorageModePrivate);
    auto device = Backend::State<DeviceType::Metal>()->getDevice();
    texture = device->newTexture(textureDesc);
    JST_ASSERT(texture, "Failed to create texture.");

    auto samplerDesc = MTL::SamplerDescriptor::alloc()->init();
    samplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    samplerState = device->newSamplerState(samplerDesc);
    samplerDesc->release();

    if (config.buffer) {
        JST_CHECK(fill());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[METAL] Destroying texture.");

    if (samplerState) {
        samplerState->release();
        samplerState = nullptr;
    }

    if (texture) {
        texture->release();
        texture = nullptr;
    }

    return Result::SUCCESS;
}

Result Implementation::underlyingDump(uint8_t* output) const {
    if (!texture) {
        JST_ERROR("[METAL] Can't dump uninitialized texture.");
        return Result::ERROR;
    }

    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    auto device = Backend::State<DeviceType::Metal>()->getDevice();
    const U64 rowByteSize = config.size.x * pixelByteSize();
    const U64 bufferByteSize = rowByteSize * config.size.y;

    auto buffer = NS::TransferPtr(device->newBuffer(bufferByteSize, MTL::ResourceStorageModeShared));
    JST_ASSERT(buffer, "[METAL] Failed to create texture dump buffer.");

    auto commandQueue = NS::TransferPtr(device->newCommandQueue());
    JST_ASSERT(commandQueue, "[METAL] Failed to create texture dump command queue.");

    auto commandBuffer = commandQueue->commandBuffer();
    JST_ASSERT(commandBuffer, "[METAL] Failed to create texture dump command buffer.");

    auto blitEncoder = commandBuffer->blitCommandEncoder();
    JST_ASSERT(blitEncoder, "[METAL] Failed to create texture dump blit encoder.");

    blitEncoder->copyFromTexture(texture, 0, 0,
                                 MTL::Origin(0, 0, 0),
                                 MTL::Size(config.size.x, config.size.y, 1),
                                 buffer.get(), 0, rowByteSize, bufferByteSize);
    blitEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    JST_ASSERT(commandBuffer->status() == MTL::CommandBufferStatusCompleted,
               "[METAL] Texture dump command buffer did not complete.");

    std::memcpy(output, buffer->contents(), bufferByteSize);

    return Result::SUCCESS;
}

MTL::PixelFormat Implementation::ConvertPixelFormat(const PixelFormat& pfmt,
                                                    const PixelType& ptype) {
    if (pfmt == PixelFormat::RED && ptype == PixelType::F32) {
        return MTL::PixelFormatR32Float;
    }

    if (pfmt == PixelFormat::RED && ptype == PixelType::UI8) {
        return MTL::PixelFormatR8Unorm;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::F32) {
        return MTL::PixelFormatRGBA32Float;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::UI8) {
        return MTL::PixelFormatRGBA8Unorm;
    }

    JST_FATAL("Can't convert pixel format.");
    throw Result::FATAL;
}

}  // namespace Jetstream::Render
