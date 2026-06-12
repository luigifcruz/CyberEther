#ifndef JETSTREAM_DOMAINS_IO_FBH_READER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_FBH_READER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct FbhReader : public Module::Config {
    std::string filepath  = "";
    U64         batchSize = 256;
    bool        loop      = true;
    bool        playing   = true;

    JST_MODULE_TYPE(fbh_reader);
    JST_MODULE_PARAMS(filepath, batchSize, loop, playing);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FBH_READER_MODULE_HH
