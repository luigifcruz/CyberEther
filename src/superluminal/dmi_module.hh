#ifndef JETSTREAM_SUPERLUMINAL_DMI_MODULE_HH
#define JETSTREAM_SUPERLUMINAL_DMI_MODULE_HH

#include <string>

#include "jetstream/module.hh"

namespace Jetstream {

#define JST_DYNAMIC_MEMORY_IMPORT_CPU(MACRO) \
    MACRO(DynamicMemoryImport, CPU, CF32) \
    MACRO(DynamicMemoryImport, CPU, F32) \
    MACRO(DynamicMemoryImport, CPU, CI8)

template<Device D, typename T = CF32>
class DynamicMemoryImport : public Module, public Compute {
 public:
    // Configuration

    struct Config {
        Tensor<D, T> buffer;

        JST_SERDES(buffer);
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
        Tensor<D, T> buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();

 private:
    JST_DEFINE_IO()
};

JST_DYNAMIC_MEMORY_IMPORT_CPU(JST_SPECIALIZATION);

}  // namespace Jetstream

#endif  // JETSTREAM_SUPERLUMINAL_DMI_MODULE_HH