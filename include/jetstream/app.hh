#ifndef JETSTREAM_APP_HH
#define JETSTREAM_APP_HH

#include "base.hh"

namespace Jetstream {

class Instance;

using PluginInitFn = void (*)(Instance* instance);

JETSTREAM_API int RunApp(int argc, char* argv[], PluginInitFn pluginInit = nullptr);

}  // namespace Jetstream

#endif  // JETSTREAM_APP_HH
