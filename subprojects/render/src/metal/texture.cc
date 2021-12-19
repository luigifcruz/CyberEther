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
    assert(textureDesc);
    textureDesc->setUsage(MTL::TextureUsagePixelFormatView);
    texture = inst.device->newTexture(textureDesc);
    assert(texture);

    if (cfg.buffer) {
        CHECK(this->fill());
    }

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Texture::destroy() {
    texture->release();

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
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
    //fmt::print("{} {} {} {}\n", xo, yo, w, h);
    auto region = MTL::Region::Make2D(xo, yo, w, h);
    texture->replaceRegion(region, 0, cfg.buffer, 4 * w);

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Texture::pour() {
    // TODO: Implement it.
    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
