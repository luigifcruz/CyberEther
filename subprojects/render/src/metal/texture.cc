#include "render/metal/texture.hpp"

namespace Render {

Metal::Texture::Texture(const Config& config, const Metal& instance)
         : Render::Texture(config), instance(instance) {
}

Result Metal::Texture::create() {
    pixelFormat = Metal::convertPixelFormat(config.pfmt, config.ptype);

    auto textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
            pixelFormat, config.size.width, config.size.height, false);
    RENDER_ASSERT(textureDesc);

    textureDesc->setUsage(MTL::TextureUsagePixelFormatView);
    texture = instance.device->newTexture(textureDesc);
    RENDER_ASSERT(texture);

    textureDesc->release();

    if (config.buffer) {
        CHECK(this->fill());
    }

    return Result::SUCCESS;
}

Result Metal::Texture::destroy() {
    texture->release();

    return Result::SUCCESS;
}

// After this method, user needs to recreate the texture
bool Metal::Texture::size(const Size2D<int>& size) {
    if (size <= Size2D<int>{1, 1}) {
        return false;
    }

    if (config.size != size) {
        config.size = size;
        return true;
    }

    return false;
}

void* Metal::Texture::raw() {
    return texture;
}

Result Metal::Texture::fill() {
    return this->fillRow(0, config.size.height);
}

Result Metal::Texture::fillRow(const std::size_t& y, const std::size_t& height) {
    auto region = MTL::Region::Make2D(0, y, config.size.width, height);
    auto rowByteSize = config.size.width * getPixelByteSize(texture->pixelFormat());
    auto bufferByteOffset = rowByteSize * y;
    texture->replaceRegion(region, 0, config.buffer + bufferByteOffset, rowByteSize);

    return Result::SUCCESS;
}

Result Metal::Texture::pour() {
    // TODO: Implement it.
    return Result::SUCCESS;
}

} // namespace Render
