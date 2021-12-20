#include "render/metal/texture.hpp"

namespace Render {

Result Metal::Texture::create() {
    if (!Metal::cudaInteropSupported() && cfg.cudaInterop) {
        cfg.cudaInterop = false;
        return Result::ERROR;
    }

    pixelFormat = Metal::convertPixelFormat(cfg.pfmt, cfg.ptype);

    auto textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
            pixelFormat, cfg.size.width, cfg.size.height, false);
    RENDER_ASSERT(textureDesc);

    textureDesc->setUsage(MTL::TextureUsagePixelFormatView);
    texture = inst.device->newTexture(textureDesc);
    RENDER_ASSERT(texture);

    textureDesc->release();

    if (cfg.buffer) {
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

    if (cfg.size != size) {
        cfg.size = size;
        return true;
    }

    return false;
}

void* Metal::Texture::raw() {
    return texture;
}

Result Metal::Texture::fill() {
    return this->fill(0, 0, cfg.size.width, cfg.size.height);
}

Result Metal::Texture::fill(int yo, int xo, int w, int h) {
    auto region = MTL::Region::Make2D(xo, yo, w, h);
    auto rowByteSize = (w - xo) * getPixelByteSize(texture->pixelFormat());
    texture->replaceRegion(region, 0, cfg.buffer, rowByteSize);

    return Result::SUCCESS;
}

Result Metal::Texture::pour() {
    // TODO: Implement it.
    return Result::SUCCESS;
}

} // namespace Render
