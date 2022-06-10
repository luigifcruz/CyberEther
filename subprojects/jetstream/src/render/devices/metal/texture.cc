#include "jetstream/render/metal/texture.hh"

namespace Jetstream::Render {

using Implementation = TextureImp<Device::Metal>;

Implementation::TextureImp(const Config& config) : Texture(config) {
}

const Result Implementation::create() {
    JST_DEBUG("Creating Metal texture.");

    pixelFormat = ConvertPixelFormat(config.pfmt, config.ptype); 

    auto textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
            pixelFormat, config.size.width, config.size.height, false);
    JST_ASSERT(textureDesc);

    textureDesc->setUsage(MTL::TextureUsagePixelFormatView);
    auto device = Backend::State<Device::Metal>()->getDevice();
    texture = device->newTexture(textureDesc); 
    JST_ASSERT(texture);

    textureDesc->release();

    if (config.buffer) {
        JST_CHECK(fill());
    }

    return Result::SUCCESS;
}

const Result Implementation::destroy() {
    JST_DEBUG("Destroying Metal texture.");

    texture->release();

    return Result::SUCCESS;
}

const bool Implementation::size(const Size2D<U64>& size) {
    if (size <= Size2D<U64>{1, 1}) {
        return false;
    }

    if (config.size != size) {
        config.size = size;
        return true;
    }

    return false;
}

const Result Implementation::fill() {
    return fillRow(0, config.size.height);
}

const Result Implementation::fillRow(const U64& y, const U64& height) {
    if (height < 1) {
        return Result::SUCCESS;
    }

    auto region = MTL::Region::Make2D(0, y, config.size.width, height);
    auto rowByteSize = config.size.width * GetPixelByteSize(texture->pixelFormat());
    auto bufferByteOffset = rowByteSize * y;
    texture->replaceRegion(region, 0, config.buffer + bufferByteOffset, rowByteSize);

    return Result::SUCCESS;
}

const MTL::PixelFormat Implementation::ConvertPixelFormat(const PixelFormat& pfmt,
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

    throw Result::ERROR;
}

const U64 Implementation::GetPixelByteSize(const MTL::PixelFormat& pfmt) {
    switch (pfmt) {
        case MTL::PixelFormatR32Float:
            return 4;
        case MTL::PixelFormatR8Unorm:
            return 1;
        case MTL::PixelFormatRGBA32Float:
            return 16;
        case MTL::PixelFormatRGBA8Unorm:
            return 4;
        default:
            throw "pixel format not implemented yet";
    }
}

}  // namespace Jetstream::Render
