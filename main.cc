#include "jetstream/run.hh"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdlib>
#endif

using namespace Jetstream;

#if defined(JST_OS_LINUX)
extern "C" void CyberEtherPluginCreate(Instance* instance) __attribute__((weak));
extern "C" void CyberEtherPluginDestroy(Instance* instance) __attribute__((weak));
#endif

#if defined(JST_OS_BROWSER)
extern "C" {
EMSCRIPTEN_KEEPALIVE
void cyberether_shutdown() {
    (void)Stop();
}
}
#endif

namespace {

int CyberEtherMain(int argc, char* argv[]) {
    try {
#if defined(JST_OS_BROWSER)
        (void)argc;
        (void)argv;
        return Run();
#endif
#if defined(JST_OS_LINUX)
        return Run(argc, argv, CyberEtherPluginCreate, CyberEtherPluginDestroy);
#endif
#if defined(JST_OS_WINDOWS) || defined(JST_OS_MAC)
        return Run(argc, argv);
#endif
    } catch (const Result& status) {
        JST_ERROR("[CYBERETHER] Exception: {}", status);
        return -1;
    } catch (const std::exception& e) {
        JST_ERROR("[CYBERETHER] Exception: {}", e.what());
        return -1;
    } catch (...) {
        JST_ERROR("[CYBERETHER] Unknown exception.");
        return -1;
    }

    return 0;
}

}

#if defined(JST_OS_WINDOWS)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    return CyberEtherMain(__argc, __argv);
}
#else
int main(int argc, char* argv[]) {
    return CyberEtherMain(argc, argv);
}
#endif
