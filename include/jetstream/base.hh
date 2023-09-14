#ifndef JETSTREAM_BASE_HH
#define JETSTREAM_BASE_HH

//
// Platform Specific
// 

#ifdef __EMSCRIPTEN__
#include "emscripten.h"
#endif

//
// System Imports
//

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/instance.hh"
#include "jetstream/module.hh"
#include "jetstream/interface.hh"
#include "jetstream/parser.hh"

//
// Functional Imports
//

#include "jetstream/modules/base.hh"
#include "jetstream/bundles/base.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/viewport/base.hh"
#include "jetstream/memory/base.hh"

#endif
