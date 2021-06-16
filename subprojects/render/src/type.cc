#include "render/type.hpp"
#include "render/tools/magic_enum.hpp"

namespace Render {

void print_error(Result res, const char* func, int line, const char* file) {
    std::cerr << "Render encountered an exception (" <<  magic_enum::enum_name(res) << ") in " \
        << func << " in line " << line << " of file " << file << "." << std::endl; \
}

} // namespace Render
