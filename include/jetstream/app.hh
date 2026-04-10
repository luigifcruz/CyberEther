#ifndef JETSTREAM_APP_HH
#define JETSTREAM_APP_HH

#include "base.hh"

namespace Jetstream {

class Instance;

using PluginCreateFn = void (*)(Instance* instance);
using PluginDestroyFn = void (*)(Instance* instance);

JETSTREAM_API int RunApp(int argc,
                         char* argv[],
                         PluginCreateFn pluginCreate = nullptr,
                         PluginDestroyFn pluginDestroy = nullptr);

}  // namespace Jetstream

#endif  // JETSTREAM_APP_HH
