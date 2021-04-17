#include "render/base.hpp"

namespace Render {

template<class T>
std::shared_ptr<T> Instance::Create(Instance::Config& cfg) {
    return std::make_shared<T>(cfg);
};

template<class T>
std::shared_ptr<Program> Instance::createProgram(Program::Config& cfg) {
    auto program = std::make_shared<typename T::Program>(cfg, *state);
    programs.push_back(program);
    return program;
}

template<class T>
std::shared_ptr<Surface> Instance::createSurface(Surface::Config& cfg) {
    auto surface = std::make_shared<typename T::Surface>(cfg, *state);
    surfaces.push_back(surface);
    return surface;
}

template<class T>
std::shared_ptr<Texture> Instance::createTexture(Texture::Config& cfg) {
    auto texture = std::make_shared<typename T::Texture>(cfg, *state);
    textures.push_back(texture);
    return texture;
}

template std::shared_ptr<GLES> Instance::Create<GLES>(Instance::Config &);
template std::shared_ptr<Program> Instance::createProgram<GLES>(Program::Config&);
template std::shared_ptr<Surface> Instance::createSurface<GLES>(Surface::Config&);
template std::shared_ptr<Texture> Instance::createTexture<GLES>(Texture::Config&);

} // namespace Render
