#ifndef JETSTREAM_MODULES_FILE_HH
#define JETSTREAM_MODULES_FILE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_FILE_WRITER_CPU(MACRO) \
    MACRO(FileWriter, CPU, CF32) \
    MACRO(FileWriter, CPU, F32)

JST_SERDES_ENUM(FileFormatType, Raw);

template<Device D, typename T = CF32>
class FileWriter : public Module, public Compute {
 public:
    FileWriter();
    ~FileWriter();

    // Configuration

    struct Config {
        FileFormatType fileFormat = FileFormatType::Raw;
        std::string filepath = "";
        std::string name = "";
        std::string description = "";
        std::string author = "CyberEther User";
        F32 sampleRate = 0.0f;
        F32 centerFrequency = 0.0f;
        bool overwrite = false;
        bool recording = false;

        JST_SERDES(fileFormat, filepath, name, description, author, sampleRate, centerFrequency, overwrite, recording);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        JST_SERDES_OUTPUT();
    };

    constexpr const Output& getOutput() const {
        return output;
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

    Result recording(const bool& recording);
    constexpr const bool& recording() const {
        return config.recording;
    }

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

#ifdef JETSTREAM_MODULE_FILE_WRITER_CPU_AVAILABLE
JST_FILE_WRITER_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::FileFormatType> : ostream_formatter {};

#endif
