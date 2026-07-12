#include "jetstream/platform.hh"

#include <exception>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#elif !defined(JST_OS_BROWSER)
#include <dlfcn.h>
#endif

namespace Jetstream::Platform {

void* OpenDynamicLibrary(const std::string& path,
                         DynamicLibraryVisibility visibility,
                         std::string& error) {
    error.clear();

#if defined(JST_OS_WINDOWS)
    (void)visibility;
    try {
        const auto nativePath = PathFromUtf8(path);
        auto* handle = LoadLibraryW(nativePath.c_str());
        if (!handle) {
            error = "Windows error " + std::to_string(GetLastError());
        }
        return reinterpret_cast<void*>(handle);
    } catch (const std::exception& exception) {
        error = exception.what();
    } catch (...) {
        error = "failed to convert the UTF-8 library path";
    }
    return nullptr;
#elif defined(JST_OS_BROWSER)
    (void)path;
    (void)visibility;
    error = "dynamic libraries are not supported on this platform";
    return nullptr;
#else
    dlerror();
    const int flags = RTLD_NOW |
                      (visibility == DynamicLibraryVisibility::Global ? RTLD_GLOBAL : RTLD_LOCAL);
    void* handle = dlopen(path.c_str(), flags);
    if (!handle) {
        if (const char* loaderError = dlerror()) {
            error = loaderError;
        } else {
            error = "unknown dynamic library error";
        }
    }
    return handle;
#endif
}

void CloseDynamicLibrary(void* handle) {
    if (!handle) {
        return;
    }

#if defined(JST_OS_WINDOWS)
    (void)FreeLibrary(reinterpret_cast<HMODULE>(handle));
#elif !defined(JST_OS_BROWSER)
    (void)dlclose(handle);
#endif
}

void* LoadDynamicLibrarySymbol(void* handle, const char* symbol, std::string& error) {
    error.clear();

#if defined(JST_OS_WINDOWS)
    auto* address = GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol);
    if (!address) {
        error = "Windows error " + std::to_string(GetLastError());
    }
    return reinterpret_cast<void*>(address);
#elif defined(JST_OS_BROWSER)
    (void)handle;
    (void)symbol;
    error = "dynamic libraries are not supported on this platform";
    return nullptr;
#else
    dlerror();
    void* address = dlsym(handle, symbol);
    if (const char* loaderError = dlerror()) {
        error = loaderError;
        return nullptr;
    }
    return address;
#endif
}

}  // namespace Jetstream::Platform
