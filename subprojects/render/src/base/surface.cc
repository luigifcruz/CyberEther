#include "render/base/surface.hpp"

namespace Render {

Result Surface::bind(std::shared_ptr<Program> program) {
    programs.push_back(program);

    return Result::SUCCESS;
}

} // namespace Render

