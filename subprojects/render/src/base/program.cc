#include "render/base/program.hpp"

namespace Render {

Result Program::bind(std::shared_ptr<Texture> texture) {
    textures.push_back(texture);

    return Result::SUCCESS;
}

} // namespace Render
