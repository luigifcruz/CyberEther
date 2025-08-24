#ifndef JETSTREAM_BASE_HH
#define JETSTREAM_BASE_HH

#include "macros.hh"

//
// Platform Specific
//

#ifdef JST_OS_BROWSER
#include "emscripten.h"
#include "emscripten/eventloop.h"
#endif

//
// System Imports
//

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/instance.hh"
#include "jetstream/module.hh"
#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/parser.hh"
#include "jetstream/benchmark.hh"

//
// Functional Imports
//

#include "jetstream/modules/base.hh"
#include "jetstream/blocks/base.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/viewport/base.hh"
#include "jetstream/memory/base.hh"

#endif
