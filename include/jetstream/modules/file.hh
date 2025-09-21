#ifndef JETSTREAM_MODULES_FILE_HH
#define JETSTREAM_MODULES_FILE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream {

JST_SERDES_ENUM(FileFormatType, Raw);

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::FileFormatType> : ostream_formatter {};

#endif
