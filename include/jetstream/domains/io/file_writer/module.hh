#ifndef JETSTREAM_DOMAINS_IO_FILE_WRITER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_FILE_WRITER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct FileWriter : public Module::Config {
    std::string filepath = "";
    std::string fileFormat = "raw";
    bool overwrite = false;
    bool recording = false;

    JST_MODULE_TYPE(file_writer);
    JST_MODULE_PARAMS(filepath, fileFormat, overwrite, recording);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FILE_WRITER_MODULE_HH
