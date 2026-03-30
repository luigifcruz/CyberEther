#ifndef JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_IMPL_HH

#include <fstream>
#include <filesystem>

#include <jetstream/domains/io/file_reader/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct FileReaderImpl : public Module::Impl, public DynamicConfig<FileReader> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    U64 getCurrentPosition() const;
    U64 getFileSize() const;

 protected:
    Tensor buffer;

    std::ifstream dataFile;
    std::filesystem::path filePath;
    Tools::Snapshot<U64> fileSize{0};
    Tools::Snapshot<U64> currentPosition{0};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FILE_READER_MODULE_IMPL_HH
