#include "jetstream/type.hpp"
#include "jetstream/tools/magic_enum.hpp"

namespace Jetstream {

void print_error(Result res, const char* func, int line, const char* file) {
    std::cerr << "Jetstream encountered an exception (" <<  magic_enum::enum_name(res) << ") in " \
        << func << " in line " << line << " of file " << file << "." << std::endl; \
}

#ifdef JETSTREAM_CUDA_AVAILABLE
void cuda_print_error(cudaError_t res, const char* func, int line, const char* file) {
    std::cerr << "CUDA encountered an exception (" \
        << std::string(magic_enum::enum_name(res)) << ") in " \
        << func << " in line " << line << " of file " \
        << file << "." << std::endl; \
}
#endif

} // namespace Jetstream
