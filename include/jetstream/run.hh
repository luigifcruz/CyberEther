#ifndef JETSTREAM_RUN_HH
#define JETSTREAM_RUN_HH

#include "base.hh"

namespace Jetstream {

#if defined(JST_OS_BROWSER)
JETSTREAM_API int Run();
JETSTREAM_API int Stop();
#endif

#if defined(JST_OS_LINUX) || defined(JST_OS_WINDOWS) || defined(JST_OS_MAC)
class Instance;

using PluginCreateFn = void (*)(Instance* instance);
using PluginDestroyFn = void (*)(Instance* instance);

JETSTREAM_API int Run(int argc,
                      char* argv[],
                      PluginCreateFn pluginCreate = nullptr,
                      PluginDestroyFn pluginDestroy = nullptr);
#endif

}  // namespace Jetstream

#endif  // JETSTREAM_RUN_HH
