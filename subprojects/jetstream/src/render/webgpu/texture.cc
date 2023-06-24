#include "jetstream/render/webgpu/texture.hh"

namespace Jetstream::Render {

using Implementation = TextureImp<Device::WebGPU>;

Implementation::TextureImp(const Config& config) : Texture(config) {
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating texture.");

    textureFormat = ConvertPixelFormat(config.pfmt, config.ptype); 

    

    auto textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
            pixelFormat, config.size.width, config.size.height, false);
    JST_ASSERT(textureDesc);

    textureDesc->setUsage(MTL::TextureUsagePixelFormatView | 
                          MTL::TextureUsageRenderTarget |
                          MTL::TextureUsageShaderRead);
    auto device = Backend::State<Device::Metal>()->getDevice();
    texture = device->newTexture(textureDesc); 
    JST_ASSERT(texture);

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
    JST_DEBUG("[WebGPU] Destroying texture.");

    texture->Destroy();

    return Result::SUCCESS;
}

bool Implementation::size(const Size2D<U64>& size) {
    if (size <= Size2D<U64>{1, 1}) {
        return false;
    }

    if (config.size != size) {
        config.size = size;
        return true;
    }

    return false;
}

Result Implementation::fill() {
    return fillRow(0, config.size.height);
}

Result Implementation::fillRow(const U64& y, const U64& height) {
    if (height < 1) {
        return Result::SUCCESS;
    }

    // TODO: implement this.

    JST_ERROR("not implemented: texture")

    // auto region = MTL::Region::Make2D(0, y, config.size.width, height);
    // auto rowByteSize = config.size.width * GetPixelByteSize(texture->pixelFormat());
    // auto bufferByteOffset = rowByteSize * y;
    // texture->replaceRegion(region, 0, config.buffer + bufferByteOffset, rowByteSize);

    return Result::SUCCESS;
}

wgpu::TextureFormat Implementation::ConvertPixelFormat(const PixelFormat& pfmt,
                                                       const PixelType& ptype) {
    if (pfmt == PixelFormat::RED && ptype == PixelType::F32) {
        return wgpu::TextureFormat::R32Float;
    }

    if (pfmt == PixelFormat::RED && ptype == PixelType::UI8) {
        return wgpu::TextureFormat::R8Unorm;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::F32) {
        return wgpu::TextureFormat::RGBA32Float;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::UI8) {
        return wgpu::TextureFormat::RGBA8Unorm;
    }

    JST_FATAL("Can't convert pixel format.");
    throw Result::ERROR;
}

U64 Implementation::GetPixelByteSize(const wgpu::TextureFormat& pfmt) {
    switch (pfmt) {
        case wgpu::TextureFormat::R32Float:
            return 4;
        case wgpu::TextureFormat::R8Unorm:
            return 1;
        case wgpu::TextureFormat::RGBA32Float:
            return 16;
        case wgpu::TextureFormat::RGBA8Unorm:
            return 4;
        default:
            JST_FATAL("Pixel format not implemented yet.");
            throw Result::ERROR;
    }
}

}  // namespace Jetstream::Render
