#ifndef JETSTREAM_VIEWPORT_BASE_HH
#define JETSTREAM_VIEWPORT_BASE_HH

#include "jetstream/viewport/generic.hh"

#ifdef JETSTREAM_VIEWPORT_MACOS_AVAILABLE
#include "jetstream/viewport/providers/macos.hh"
#endif

#ifdef JETSTREAM_VIEWPORT_IOS_AVAILABLE
#include "jetstream/viewport/providers/ios.hh"
#endif

#ifdef JETSTREAM_VIEWPORT_LINUX_AVAILABLE
#include "jetstream/viewport/providers/linux.hh"
#endif

#endif
