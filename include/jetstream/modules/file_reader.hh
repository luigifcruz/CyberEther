#ifndef JETSTREAM_MODULES_FILE_READER_HH
#define JETSTREAM_MODULES_FILE_READER_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/modules/file.hh"
#include "jetstream/memory2/tensor.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_FILE_READER_CPU(MACRO) \
    MACRO(FileReader, CPU, CF32) \
    MACRO(FileReader, CPU, F32)

template<Device D, typename T = CF32>
class FileReader : public Module, public Compute {
 public:
    FileReader();
    ~FileReader();

    // Configuration

    struct Config {
        FileFormatType fileFormat = FileFormatType::Raw;
        std::string filepath = "";
        bool playing = false;
        bool loop = true;
        std::vector<U64> shape = {8192};

        JST_SERDES(fileFormat, filepath, playing, loop, shape);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES_INPUT();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    constexpr Taint taint() const {
        // TODO: Implement discontiguous support for this module.
        return Taint::CLEAN;
    }

    void info() const final;

    // Constructor

    Result create();
    Result destroy();

    // Miscellaneous

    Result playing(const bool& playing);
    constexpr const bool& playing() const {
        return config.playing;
    }

    U64 getFileSize() const;
    U64 getCurrentPosition() const;

 protected:
    Result createCompute(const Context& ctx) final;
    Result destroyCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    struct GImpl;
    std::unique_ptr<GImpl> gimpl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_FILE_READER_CPU_AVAILABLE
JST_FILE_READER_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
