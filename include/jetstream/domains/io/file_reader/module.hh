#ifndef JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct FileReader : public Module::Config {
    std::string filepath = "";
    std::string fileFormat = "raw";
    std::string dataType = "CF32";
    U64 batchSize = 8192;
    bool loop = true;
    bool playing = true;

    JST_MODULE_TYPE(file_reader);
    JST_MODULE_PARAMS(filepath, fileFormat, dataType, batchSize, loop, playing);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_HH
