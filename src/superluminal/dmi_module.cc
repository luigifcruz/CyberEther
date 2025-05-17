#include "dmi_module.hh"

namespace Jetstream {

template<Device D, typename T>
void DynamicMemoryImport<D, T>::info() const {
    JST_DEBUG("  None");
}

template<Device D, typename T>
Result DynamicMemoryImport<D, T>::create() {
    JST_DEBUG("[SUPERLUMINAL] Initializing Dynamic Memory Import module.");
    JST_INIT_IO();

    output.buffer = config.buffer;

    return Result::SUCCESS;
}

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
JST_DYNAMIC_MEMORY_IMPORT_CPU(JST_INSTANTIATION)
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
JST_DYNAMIC_MEMORY_IMPORT_CUDA(JST_INSTANTIATION)
#endif

}  // namespace Jetstream
