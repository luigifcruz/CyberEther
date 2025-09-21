#include "jetstream/render/devices/webgpu/texture.hh"


namespace Jetstream::Render {

using Implementation = TextureImp<Device::WebGPU>;

Implementation::TextureImp(const Config& config) : Texture(config) {
}

Result Implementation::create() {
    JST_DEBUG("[WebGPU] Creating texture.");

    auto device = Backend::State<Device::WebGPU>()->getDevice();

    textureFormat = ConvertPixelFormat(config.pfmt, config.ptype);

    WGPUTextureDescriptor textureDescriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    textureDescriptor.size = WGPUExtent3D{
        static_cast<uint32_t>(config.size.x),
        static_cast<uint32_t>(config.size.y),
        1u
    };
    textureDescriptor.format = textureFormat;
    textureDescriptor.usage = WGPUTextureUsage_RenderAttachment |
                               WGPUTextureUsage_CopyDst |
                               WGPUTextureUsage_TextureBinding;
    textureDescriptor.dimension = WGPUTextureDimension_2D;
    texture = wgpuDeviceCreateTexture(device, &textureDescriptor);

    WGPUTextureViewDescriptor viewDescriptor = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    viewDescriptor.format = textureFormat;
    viewDescriptor.dimension = WGPUTextureViewDimension_2D;
    textureView = wgpuTextureCreateView(texture, &viewDescriptor);

    // Using Nearest because 'float32-filterable' is not yet widely supported.
    WGPUSamplerDescriptor samplerDescriptor = WGPU_SAMPLER_DESCRIPTOR_INIT;
    samplerDescriptor.magFilter = WGPUFilterMode_Nearest;
    samplerDescriptor.minFilter = WGPUFilterMode_Nearest;
    sampler = wgpuDeviceCreateSampler(device, &samplerDescriptor);

    textureBindingLayout = WGPU_TEXTURE_BINDING_LAYOUT_INIT;
    textureBindingLayout.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    textureBindingLayout.viewDimension = WGPUTextureViewDimension_2D;

    samplerBindingLayout = WGPU_SAMPLER_BINDING_LAYOUT_INIT;
    samplerBindingLayout.type = WGPUSamplerBindingType_NonFiltering;

    if (config.buffer) {
        JST_CHECK(fill());
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[WebGPU] Destroying texture.");

    wgpuTextureDestroy(texture);

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

    WGPUDevice device = Backend::State<Device::WebGPU>()->getDevice();
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    layout.bytesPerRow = static_cast<uint32_t>(config.size.x * GetPixelByteSize(textureFormat));
    layout.rowsPerImage = static_cast<uint32_t>(config.size.y);

    WGPUExtent3D extent = {};
    extent.width = static_cast<uint32_t>(config.size.x);
    extent.height = static_cast<uint32_t>(height);
    extent.depthOrArrayLayers = 1u;

    WGPUTexelCopyTextureInfo copyTexture = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    copyTexture.texture = texture;
    copyTexture.mipLevel = 0;
    copyTexture.origin = WGPUOrigin3D{0, static_cast<uint32_t>(y), 0};
    copyTexture.aspect = WGPUTextureAspect_All;

    wgpuQueueWriteTexture(queue,
                          &copyTexture,
                          static_cast<const void*>(config.buffer),
                          layout.bytesPerRow * layout.rowsPerImage,
                          &layout,
                          &extent);

    return Result::SUCCESS;
}

WGPUTextureFormat Implementation::ConvertPixelFormat(const PixelFormat& pfmt,
                                                     const PixelType& ptype) {
    if (pfmt == PixelFormat::RED && ptype == PixelType::F32) {
        return WGPUTextureFormat_R32Float;
    }

    if (pfmt == PixelFormat::RED && ptype == PixelType::UI8) {
        return WGPUTextureFormat_R8Unorm;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::F32) {
        return WGPUTextureFormat_RGBA32Float;
    }

    if (pfmt == PixelFormat::RGBA && ptype == PixelType::UI8) {
        return WGPUTextureFormat_RGBA8Unorm;
    }

    JST_FATAL("[WebGPU] Can't convert pixel format.");
    throw Result::ERROR;
}

U64 Implementation::GetPixelByteSize(const WGPUTextureFormat& pfmt) {
    switch (pfmt) {
        case WGPUTextureFormat_R32Float:
            return 4;
        case WGPUTextureFormat_R8Unorm:
            return 1;
        case WGPUTextureFormat_RGBA32Float:
            return 16;
        case WGPUTextureFormat_RGBA8Unorm:
            return 4;
        default:
            JST_FATAL("[WebGPU] Pixel format not implemented yet.");
            throw Result::ERROR;
    }
}

}  // namespace Jetstream::Render
