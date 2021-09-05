#include "jetstream/helpers.hpp"

#ifdef JETSTREAM_CUDA_AVAILABLE
void jst_cuda_print_error(cudaError_t result, const char* func, int line, const char* file) {
    std::cerr << "CUDA encountered an exception (" \
        << std::string(cudaGetErrorString(result)) << ") in " \
        << func << " in line " <<  line<< " of file " \
        << file << "." << std::endl; \
}
#endif
