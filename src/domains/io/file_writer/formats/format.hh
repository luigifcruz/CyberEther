#ifndef JETSTREAM_DOMAINS_IO_FILE_WRITER_FORMATS_FORMAT_HH
#define JETSTREAM_DOMAINS_IO_FILE_WRITER_FORMATS_FORMAT_HH

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <jetstream/memory/tensor.hh>
#include <jetstream/types.hh>

namespace Jetstream::Modules {

class FileWriterFormat {
 public:
    struct Option {
        std::string value;
        std::string label;
    };

    class Raw;
    class Sigmf;
    class Wav;

    virtual ~FileWriterFormat() = default;

    static const std::vector<Option>& Options();
    static std::unique_ptr<FileWriterFormat> Create(const std::string& name);

    virtual std::filesystem::path resolve(const std::filesystem::path& path);
    virtual std::vector<std::filesystem::path> companions() const;
    virtual Result check(const Tensor& input) const;
    virtual Result start(std::ofstream& file, const Tensor& input);
    virtual Result track(const Tensor& input, U64 sampleIndex);
    virtual Result finish(std::ofstream& file, U64 dataBytes);

 protected:
    static std::optional<F64> NumericAttribute(const Tensor& tensor, const std::string& key);
};

class FileWriterFormat::Raw : public FileWriterFormat {};

class FileWriterFormat::Sigmf : public FileWriterFormat {
 public:
    std::filesystem::path resolve(const std::filesystem::path& path) override;
    std::vector<std::filesystem::path> companions() const override;
    Result check(const Tensor& input) const override;
    Result start(std::ofstream& file, const Tensor& input) override;
    Result track(const Tensor& input, U64 sampleIndex) override;
    Result finish(std::ofstream& file, U64 dataBytes) override;

 private:
    struct Capture {
        U64 sampleStart = 0;
        std::optional<F64> frequency;
        std::string datetime;
    };

    // Losses while recording, from the attributes a radio source publishes:
    // the position of each frame in its received stream, and its running
    // counts of samples the radio dropped and of overflows.
    struct Losses {
        bool tracked = false;
        std::optional<U64> nextPosition;
        U64 lostSamples = 0;
        std::vector<std::pair<U64, U64>> gaps;
        std::optional<U64> radioLostFirst;
        std::optional<U64> radioLostLast;
        std::vector<std::pair<U64, U64>> radioGaps;
        std::optional<U64> overflowsFirst;
        std::optional<U64> overflowsLast;
    };

    Result write() const;

    std::filesystem::path metaPath;
    std::string datatype;
    std::optional<F64> sampleRate;
    std::vector<Capture> captures;
    Losses losses;
};

class FileWriterFormat::Wav : public FileWriterFormat {
 public:
    Result check(const Tensor& input) const override;
    Result start(std::ofstream& file, const Tensor& input) override;
    Result track(const Tensor& input, U64 sampleIndex) override;
    Result finish(std::ofstream& file, U64 dataBytes) override;

 private:
    std::string header(std::optional<U64> dataBytes) const;

    U16 format = 0;
    U16 channels = 0;
    U16 bits = 0;
    std::optional<F64> sampleRate;
    std::optional<U32> frequency;
    std::chrono::system_clock::time_point startTime{};
    bool started = false;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_FILE_WRITER_FORMATS_FORMAT_HH
