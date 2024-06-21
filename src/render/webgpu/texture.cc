#include "jetstream/render/webgpu/texture.hh"

namespace Jetstream::Render {

using Implementation = TextureImp<Device::WebGPU>;

Implementation::TextureImp(const Config& config) : Texture(config) {
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating texture.");

    auto device = Backend::State<Device::WebGPU>()->getDevice();

    textureFormat = ConvertPixelFormat(config.pfmt, config.ptype); 
        
    wgpu::TextureDescriptor textureDescriptor{};
    textureDescriptor.size = wgpu::Extent3D{static_cast<U32>(config.size.x),
                                            static_cast<U32>(config.size.y)};
    textureDescriptor.format = textureFormat;
    textureDescriptor.usage = wgpu::TextureUsage::RenderAttachment |
                              wgpu::TextureUsage::CopyDst | 
                              wgpu::TextureUsage::TextureBinding;
    texture = device.CreateTexture(&textureDescriptor);

    wgpu::TextureViewDescriptor viewDescriptor{};
    viewDescriptor.format = textureFormat;
    viewDescriptor.dimension = wgpu::TextureViewDimension::e2D;
    textureView = texture.CreateView(&viewDescriptor);

    // Using Nearest because 'float32-filterable' is not yet widely supported.
    wgpu::SamplerDescriptor samplerDescriptor{};
    samplerDescriptor.magFilter = wgpu::FilterMode::Nearest;
    samplerDescriptor.minFilter = wgpu::FilterMode::Nearest;
    sampler = device.CreateSampler(&samplerDescriptor);

    textureBindingLayout = {};
    textureBindingLayout.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
    textureBindingLayout.viewDimension = wgpu::TextureViewDimension::e2D;

    samplerBindingLayout = {};
    samplerBindingLayout.type = wgpu::SamplerBindingType::NonFiltering;
    
    if (config.buffer) {
        JST_CHECK(fill());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying texture.");

    texture.Destroy();

    return Result::SUCCESS;
}

bool Implementation::size(const Extent2D<U64>& size) {
    if (size <= Extent2D<U64>{1, 1}) {
        return false;
    }

    if (config.size != size) {
        config.size = size;
        return true;
    }

    return false;
}

Result Implementation::fill() {
    return fillRow(0, config.size.y);
}

Result Implementation::fillRow(const U64& y, const U64& height) {
    if (height < 1) {
        return Result::SUCCESS;
    }

    auto& device = Backend::State<Device::WebGPU>()->getDevice();

    wgpu::TextureDataLayout layout;
    layout.bytesPerRow = config.size.x * GetPixelByteSize(textureFormat);
    layout.rowsPerImage = config.size.y;

    wgpu::Extent3D extent;
    extent.width = config.size.x;
    extent.height = height;

    wgpu::ImageCopyTexture copyTexture;
    copyTexture.origin = {0, static_cast<U32>(y), 0};
    copyTexture.texture = texture;
        
    device.GetQueue().WriteTexture(&copyTexture, (uint8_t*)config.buffer, layout.bytesPerRow * layout.rowsPerImage, &layout, &extent);

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

    JST_FATAL("[WebGPU] Can't convert pixel format.");
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
            JST_FATAL("[WebGPU] Pixel format not implemented yet.");
            throw Result::ERROR;
    }
}

}  // namespace Jetstream::Render
