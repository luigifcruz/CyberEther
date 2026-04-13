#ifndef JETSTREAM_APP_HH
#define JETSTREAM_APP_HH

#include "base.hh"

namespace Jetstream {

#if defined(JST_OS_BROWSER)
JETSTREAM_API int RunAppBrowser();
JETSTREAM_API int StopAppBrowser();
#endif

#if defined(JST_OS_LINUX) || defined(JST_OS_WINDOWS) || defined(JST_OS_MAC)
class Instance;

using PluginCreateFn = void (*)(Instance* instance);
using PluginDestroyFn = void (*)(Instance* instance);

JETSTREAM_API int RunAppNative(int argc,
                               char* argv[],
                               PluginCreateFn pluginCreate = nullptr,
                               PluginDestroyFn pluginDestroy = nullptr);
#endif

}  // namespace Jetstream

#endif  // JETSTREAM_APP_HH
