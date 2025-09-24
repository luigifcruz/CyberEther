#ifndef JETSTREAM_MODULES_SPECTOGRAM_HH
#define JETSTREAM_MODULES_SPECTOGRAM_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory2/tensor.hh"
#include "jetstream/render/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_SPECTROGRAM_CPU(MACRO) \
    MACRO(Spectrogram, CPU, F32)

#define JST_SPECTROGRAM_METAL(MACRO) \
    MACRO(Spectrogram, Metal, F32)

#define JST_SPECTROGRAM_CUDA(MACRO) \
    MACRO(Spectrogram, CUDA, F32)

template<Device D, typename T = F32>
class Spectrogram : public Module, public Compute, public Present {
 public:
    Spectrogram();
    ~Spectrogram();

    // Configuration

    struct Config {
        U64 height = 256;
        Extent2D<U64> viewSize = {512, 384};

        JST_SERDES(height, viewSize);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        mem2::Tensor buffer;

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

    void info() const final;

    // Constructor

    Result create();

    // Miscellaneous

    constexpr const Extent2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Extent2D<U64>& viewSize(const Extent2D<U64>& viewSize);

    Render::Texture& getTexture();

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

    Result createPresent() final;
    Result present() final;
    Result destroyPresent() final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    struct GImpl;
    std::unique_ptr<GImpl> gimpl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE
JST_SPECTROGRAM_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_CUDA_AVAILABLE
JST_SPECTROGRAM_CUDA(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_SPECTROGRAM_METAL_AVAILABLE
JST_SPECTROGRAM_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
