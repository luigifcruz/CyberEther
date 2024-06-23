#ifndef JETSTREAM_MODULES_SPEECH_RECOGNITION_HH
#define JETSTREAM_MODULES_SPEECH_RECOGNITION_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/assets.hh"
#include "jetstream/compute/graph/base.hh"

#include <whisper.h>

namespace Jetstream {

#define JST_SPEECH_RECOGNITION_CPU(MACRO) \
    MACRO(SpeechRecognition, CPU, F32)

template<Device D, typename T = F32>
class SpeechRecognition : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        JST_SERDES();
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

    constexpr const std::string& getTextBuffer() const {
        return textBuffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

 private:
    // TODO: Remove backend specific code from header in favor of `pimpl->`.
    struct whisper_context* ctx;
    whisper_full_params wparams;
    std::vector<whisper_token> promptTokens;
    std::string textBuffer;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_SPEECH_RECOGNITION_CPU_AVAILABLE
JST_SPEECH_RECOGNITION_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
