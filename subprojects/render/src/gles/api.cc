#include "gles/api.hpp"
#include "gles/instance.hpp"
#include "gles/program.hpp"
#include "gles/surface.hpp"

namespace Render {

GLES::GLES() {
    state = std::make_unique<GLES::State>();
}

std::shared_ptr<GLES::Instance> GLES::createInstance(Render::Instance::Config& cfg) {
    if (!instance) {
        instance = std::make_shared<Instance>(cfg, *state.get());
    }
    return instance;
}

std::shared_ptr<GLES::Program> GLES::createProgram(Render::Program::Config& cfg) {
    return instance->_createProgram(cfg, *state.get());
}


Result GLES::getError(std::string func, std::string file, int line) {
    int error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cout << "[OPENGL] GL returned an error #" << error
                  << " inside function " << func << " @ "
                  << file << ":" << line << std::endl;
        return Result::RENDER_BACKEND_ERROR;
    }
    return Result::SUCCESS;
}

} // namespace Render
