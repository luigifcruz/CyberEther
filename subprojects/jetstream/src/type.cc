#include "jetstream/type.hpp"
#include "jetstream/tools/magic_enum.hpp"

namespace Jetstream {

void print_error(Result res, const char* func, int line, const char* file) {
    std::cerr << "Jetstream encountered an exception (" <<  magic_enum::enum_name(res) << ") in " \
        << func << " in line " << line << " of file " << file << "." << std::endl; \
}

} // namespace Jetstream
