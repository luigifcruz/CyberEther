#ifndef JETSTREAM_DOMAINS_IO_FILE_READER_FORMATS_FORMAT_HH
#define JETSTREAM_DOMAINS_IO_FILE_READER_FORMATS_FORMAT_HH

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <jetstream/memory/types.hh>
#include <jetstream/types.hh>

namespace Jetstream::Modules {

class FileReaderFormat {
 public:
    struct Option {
        std::string value;
        std::string label;
    };

    struct Recording {
        struct Capture {
            U64 sampleStart = 0;
            std::optional<F64> frequency;
        };

        std::filesystem::path dataPath;
        DataType dataType = DataType::None;
        U64 headerBytes = 0;
        U64 trailingBytes = 0;
        std::optional<F64> sampleRate;
        std::vector<Capture> captures;
    };

    class Raw;
    class Sigmf;
    class Wav;

    virtual ~FileReaderFormat() = default;

    static const std::vector<Option>& Options();
    static std::unique_ptr<FileReaderFormat> Create(const std::string& name);

    virtual bool describesSamples() const;
    virtual Result load(const std::string& filepath, Recording& recording) const = 0;
};

class FileReaderFormat::Raw : public FileReaderFormat {
 public:
    bool describesSamples() const override;
    Result load(const std::string& filepath, Recording& recording) const override;
};

class FileReaderFormat::Sigmf : public FileReaderFormat {
 public:
    Result load(const std::string& filepath, Recording& recording) const override;
};

class FileReaderFormat::Wav : public FileReaderFormat {
 public:
    Result load(const std::string& filepath, Recording& recording) const override;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FILE_READER_FORMATS_FORMAT_HH
