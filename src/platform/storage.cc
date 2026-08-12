#include "jetstream/platform.hh"

#if defined(JST_OS_BROWSER)
#include <emscripten/wasmfs.h>
#endif

namespace Jetstream::Platform {

Result InitializePersistentStorage() {
#if defined(JST_OS_BROWSER)
    backend_t opfs = wasmfs_create_opfs_backend();
    const int status = wasmfs_create_directory("/storage", 0777, opfs);
    JST_DEBUG("OPFS mount on /storage: {}", status == 0 ? "OK" : "FAILED");
    return status == 0 ? Result::SUCCESS : Result::ERROR;
#else
    return Result::SUCCESS;
#endif
}

}  // namespace Jetstream::Platform
