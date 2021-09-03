#include "gadget/base.hpp"
#include "gadget/tools/magic_enum.hpp"

namespace Gadget {

#ifdef GADGET_HAS_CUDA
void cuda_print_error(cudaError_t res, const char* func, int line, const char* file) {
    std::cerr << "CUDA encountered an exception (" \
        << std::string(magic_enum::enum_name(res)) << ") in " \
        << func << " in line " << line << " of file " \
        << file << "." << std::endl; \
}
#endif

} // namespace Gadget
