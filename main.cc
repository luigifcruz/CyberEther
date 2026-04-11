#include "jetstream/app.hh"

using namespace Jetstream;

#ifdef JST_OS_LINUX
extern "C" void CyberEtherPluginCreate(Instance* instance) __attribute__((weak));
extern "C" void CyberEtherPluginDestroy(Instance* instance) __attribute__((weak));
#endif

int main(int argc, char* argv[]) {
#ifdef JST_OS_LINUX
    const Result res = RunApp(argc, argv, CyberEtherPluginCreate, CyberEtherPluginDestroy);
#else
    const Result res = RunApp(argc, argv);
#endif

    if (res != Result::SUCCESS) {
        return 1;
    }

    return 0;
}
