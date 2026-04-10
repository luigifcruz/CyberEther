#include "jetstream/app.hh"

#ifdef JST_OS_LINUX
extern "C" void CyberEtherPluginCreate(Jetstream::Instance* instance) __attribute__((weak));
extern "C" void CyberEtherPluginDestroy(Jetstream::Instance* instance) __attribute__((weak));
#endif

int main(int argc, char* argv[]) {
#ifdef JST_OS_LINUX
    return Jetstream::RunApp(argc, argv, CyberEtherPluginCreate, CyberEtherPluginDestroy);
#else
    return Jetstream::RunApp(argc, argv);
#endif
}
