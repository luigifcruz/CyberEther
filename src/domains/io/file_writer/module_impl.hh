#ifndef JETSTREAM_DOMAINS_IO_FILE_WRITER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_FILE_WRITER_MODULE_IMPL_HH

#include <fstream>
#include <filesystem>

#include <jetstream/domains/io/file_writer/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct FileWriterImpl : public Module::Impl, public DynamicConfig<FileWriter> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    U64 getBytesWritten() const;

 protected:
    std::ofstream dataFile;
    std::filesystem::path filePath;
    Tools::Snapshot<U64> bytesWritten{0};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FILE_WRITER_MODULE_IMPL_HH
