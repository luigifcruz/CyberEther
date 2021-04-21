#include "render/base.hpp"

namespace Render {

template<class T>
std::shared_ptr<T> Instance::Create(Instance::Config& cfg) {
    return std::make_shared<T>(cfg);
};

template<class T>
std::shared_ptr<Program> Instance::create(Program::Config& cfg) {
    return std::make_shared<typename T::Program>(cfg, *state);
}

template<class T>
std::shared_ptr<Surface> Instance::create(Surface::Config& cfg) {
    auto surface = std::make_shared<typename T::Surface>(cfg, *state);
    this->cfg.surfaces.push_back(surface);
    return surface;
}

template<class T>
std::shared_ptr<Texture> Instance::create(Texture::Config& cfg) {
    return std::make_shared<typename T::Texture>(cfg, *state);
}

template<class T>
std::shared_ptr<Vertex> Instance::create(Vertex::Config& cfg) {
    return std::make_shared<typename T::Vertex>(cfg, *state);
}

template std::shared_ptr<GLES> Instance::Create<GLES>(Instance::Config &);
template std::shared_ptr<Program> Instance::create<GLES>(Program::Config&);
template std::shared_ptr<Surface> Instance::create<GLES>(Surface::Config&);
template std::shared_ptr<Texture> Instance::create<GLES>(Texture::Config&);
template std::shared_ptr<Vertex> Instance::create<GLES>(Vertex::Config&);

} // namespace Render
