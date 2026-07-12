#include "jetstream/platform.hh"

#include <cstdlib>

#if defined(JST_OS_BROWSER)
#include <emscripten.h>
#elif defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream::Platform {

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result OpenUrl(const std::string& url) {
    emscripten_run_script(jst::fmt::format("window.open('{}', '_blank');", url).c_str());
    return Result::SUCCESS;
}

#elif defined(JST_OS_LINUX)

Result OpenUrl(const std::string& url) {
    const auto res = system(jst::fmt::format("xdg-open ""{}""", url).c_str());
    if (res != 0) {
        JST_ERROR("Failed to open URL: {}", url);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

Result OpenUrl(const std::string& url) {
    const INT_PTR result = reinterpret_cast<INT_PTR>(
        ShellExecuteA(nullptr, nullptr, url.c_str(), nullptr, nullptr, SW_SHOW));
    if (result < 32) {
        JST_ERROR("Failed to open URL: {}", url);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

#else

Result OpenUrl(const std::string& url) {
    JST_ERROR("Opening URL is not supported in this platform.");
    return Result::ERROR;
}

#endif

}  // namespace Jetstream::Platform
