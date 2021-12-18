#include "render/metal/texture.hpp"

namespace Render {

Result Metal::Texture::create() {
    if (!Metal::cudaInteropSupported() && cfg.cudaInterop) {
        cfg.cudaInterop = false;
        return Result::ERROR;
    }

    // TODO: Change Pixel
    auto textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
            MTL::PixelFormatBGRA8Unorm, cfg.size.width, cfg.size.height, false);
    assert(textureDesc);
    textureDesc->setUsage(MTL::TextureUsagePixelFormatView);
    texture = inst.device->newTexture(textureDesc);
    assert(texture);

    fmt::print("Tex ok!\n");

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

uint Metal::Texture::raw() {
    ImGui::Begin("Lineplot");
    ImGui::Image((void*)texture, ImVec2(720, 480));
    ImGui::End();

    return (uintptr_t)texture;
}

Result Metal::Texture::fill() {
    /*
    if (cfg.cudaInterop) {
        return this->_cudaCopyToTexture(0, 0, cfg.size.width, cfg.size.height);
    }

    CHECK(this->begin());
    glTexImage2D(GL_TEXTURE_2D, 0, dfmt, cfg.size.width, cfg.size.height, 0, pfmt, ptype, cfg.buffer);
    CHECK(this->end());
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Texture::fill(int yo, int xo, int w, int h) {
    /*
    if (cfg.cudaInterop) {
        return this->_cudaCopyToTexture(yo, xo, w, h);
    }

    CHECK(this->begin());
    size_t offset = yo * w * ((cfg.ptype == PixelType::F32) ? sizeof(float) : sizeof(uint));
    glTexSubImage2D(GL_TEXTURE_2D, 0, xo, yo, w, h, pfmt, ptype, cfg.buffer + offset);
    CHECK(this->end());
    */

    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

Result Metal::Texture::pour() {
    // TBD
    return Metal::getError(__FUNCTION__, __FILE__, __LINE__);
}

} // namespace Render
