#ifndef JETSTREAM_BLOCK_FILTER_ENGINE_BASE_HH
#define JETSTREAM_BLOCK_FILTER_ENGINE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/pad.hh"
#include "jetstream/modules/fft.hh"
#include "jetstream/modules/multiply.hh"
#include "jetstream/modules/unpad.hh"
#include "jetstream/modules/overlap_add.hh"
#include "jetstream/modules/fold.hh"
#include "jetstream/modules/tensor_modifier.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class FilterEngine : public Block {
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
        Tensor<D, IT> signal;
        Tensor<D, IT> filter;

        JST_SERDES(signal, filter);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "filter-engine";
    }

    std::string name() const {
        return "Filter Engine";
    }

    std::string summary() const {
        return "Filters a signal using taps.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Filter the input signal using the provided filter taps. This block applies "
               "the filter using the overlap-add method in the frequency domain.";
    }

    // Constructor

    std::string warning() const {
        return _warning;
    }

    Result create() {
        U64 filterMaxRank = input.filter.rank() - 1;
        const U64 filterSize = input.filter.shape()[filterMaxRank];

        U64 signalMaxRank = input.signal.rank() - 1;
        const U64 signalSize = input.signal.shape()[signalMaxRank];

        JST_CHECK(instance().addModule(
            padSignal, "padSignal", {
                .size = filterSize - 1,
                .axis = signalMaxRank,
            }, {
                .unpadded = input.signal,
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            padFilter, "padFilter", {
                .size = signalSize - 1,
                .axis = filterMaxRank,
            }, {
                .unpadded = input.filter,
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            fftSignal, "fftSignal", {
                .forward = true,
            }, {
                .buffer = padSignal->getOutputPadded(),
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            fftFilter, "fftFilter", {
                .forward = true,
            }, {
                .buffer = padFilter->getOutputPadded(),
            },
            locale()
        ));

        auto multiplySignalInput = fftSignal->getOutputBuffer();

        if (input.filter.rank() == 2 && (input.signal.rank() == 1 || input.signal.rank() == 2)) {
            JST_DEBUG("Filter is 2D, adding a dimension to the signal.");

            JST_CHECK(instance().addModule(
                expandDims, "expandDims", {
                    .callback = [&](auto& mod) {
                        mod.expand_dims(signalMaxRank);
                        return Result::SUCCESS;
                    }
                }, {
                    .buffer = fftSignal->getOutputBuffer(),
                },
                locale()
            ));

            multiplySignalInput = expandDims->getOutputBuffer();
            signalMaxRank = multiplySignalInput.rank() - 1;
        }

        JST_CHECK(instance().addModule(
            multiply, "multiply", {}, {
                .factorA = multiplySignalInput,
                .factorB = fftFilter->getOutputBuffer(),
            },
            locale()
        ));

        auto ifftInput = multiply->getOutputProduct();

        if (calculateResampleHeuristics(filterSize, signalSize)) {
            JST_CHECK(instance().addModule(
                fold, "fold", {
                    .axis = std::max(filterMaxRank, signalMaxRank),
                    .offset = resamplerOffset,
                    .size = resamplerSize,
                }, {
                    .buffer = multiply->getOutputProduct(),
                },
                locale()
            ));

            ifftInput = fold->getOutputBuffer();
        }

        JST_CHECK(instance().addModule(
            ifft, "ifft", {
                .forward = false,
            }, {
                .buffer = ifftInput,
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            unpad, "unpad", {
                .size = padSize,
                .axis = std::max(filterMaxRank, signalMaxRank),
            }, {
                .padded = ifft->getOutputBuffer(),
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            overlap, "overlap", {
                .axis = std::max(filterMaxRank, signalMaxRank),
            }, {
                .buffer = unpad->getOutputUnpadded(),
                .overlap = unpad->getOutputPad(),
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, overlap->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(overlap->locale()));
        JST_CHECK(instance().eraseModule(unpad->locale()));
        JST_CHECK(instance().eraseModule(ifft->locale()));

        if (fold) {
            JST_CHECK(instance().eraseModule(fold->locale()));
        }

        JST_CHECK(instance().eraseModule(multiply->locale()));

        if (expandDims) {
            JST_CHECK(instance().eraseModule(expandDims->locale()));
        }

        JST_CHECK(instance().eraseModule(fftFilter->locale()));
        JST_CHECK(instance().eraseModule(fftSignal->locale()));
        JST_CHECK(instance().eraseModule(padFilter->locale()));
        JST_CHECK(instance().eraseModule(padSignal->locale()));

        resamplerOffset = 0;
        resamplerSize = 0;
        padSize = 0;

        return Result::SUCCESS;
    }

 private:
    std::string _warning;
    U64 resamplerOffset = 0;
    U64 resamplerSize = 0;
    U64 padSize = 0;

    std::shared_ptr<Jetstream::Pad<D, IT>> padSignal;
    std::shared_ptr<Jetstream::Pad<D, IT>> padFilter;
    std::shared_ptr<Jetstream::FFT<D, IT>> fftSignal;
    std::shared_ptr<Jetstream::FFT<D, IT>> fftFilter;
    std::shared_ptr<Jetstream::TensorModifier<D, IT>> expandDims;
    std::shared_ptr<Jetstream::Multiply<D, IT>> multiply;
    std::shared_ptr<Jetstream::Fold<D, IT>> fold;
    std::shared_ptr<Jetstream::FFT<D, IT>> ifft;
    std::shared_ptr<Jetstream::Unpad<D, IT>> unpad;
    std::shared_ptr<Jetstream::OverlapAdd<D, IT>> overlap;

    bool calculateResampleHeuristics(const U64& filterSize, const U64& signalSize) {
        // Calculate default pad size without resampling.
        padSize = filterSize - 1;

        // Check if filter has all necessary attributes.

        std::vector<std::string> dependency_keys;
        dependency_keys.push_back("sample_rate");
        dependency_keys.push_back("bandwidth");
        dependency_keys.push_back("center");

        for (const auto& key : dependency_keys) {
            if (!input.filter.attributes().contains(key)) {
                _warning = "Bypassing resampling because filter is not passing necessary attributes.";
                break;
            }
        }

        // If so, check if resampling is necessary and possible.

        if (_warning.empty()) {
            const F32& sampleRate = input.filter.attribute("sample_rate").template get<F32>();
            const F32& bandwidth = input.filter.attribute("bandwidth").template get<F32>();
            const F32& center = input.filter.attribute("center").template get<std::vector<F32>>()[0];

            const F32 resamplerRatio = sampleRate / bandwidth;

            if (resamplerRatio != std::floor(resamplerRatio)) {
                _warning = jst::fmt::format("Bypassing resampling because filter bandwidth ({:.2f} MHz) "
                                            "is not a multiple of the signal sample rate ({:.2f} MHz).",
                                            bandwidth / JST_MHZ, sampleRate / JST_MHZ);
                return false;
            }

            if ((padSize / resamplerRatio) != std::floor(padSize / resamplerRatio)) {
                _warning = jst::fmt::format("Bypassing resampling because filter tap size minus one ({}) "
                                            " is not a multiple of the resampler ratio ({}).", 
                                            padSize, resamplerRatio);
                return false;
            }

            resamplerSize = (filterSize + signalSize - 1);
            if ((resamplerSize / resamplerRatio) != std::floor(resamplerSize / resamplerRatio)) {
                _warning = jst::fmt::format("Bypassing resampling because filter tap size minus one ({}) "
                                            "plus signal size ({}) is not a multiple of the "
                                            "resampler ratio ({}).", 
                                            padSize, signalSize, resamplerRatio);
                return false;
            }

            if (center != 0.0f) {
                const F32 frequencyPerBin = sampleRate / static_cast<F32>(resamplerSize);
                const F32 centerBin = center / frequencyPerBin;

                if (centerBin != std::floor(centerBin)) {
                    _warning = jst::fmt::format("Output will be shifted by {} MHz because filter "
                                                "center frequency ({:.2f} MHz) is not a multiple of the "
                                                "frequency per bin ({} MHz).",
                                                (centerBin - std::floor(centerBin)) * frequencyPerBin / JST_MHZ, 
                                                center / JST_MHZ, 
                                                frequencyPerBin / JST_MHZ);
                }

                // TODO: Looks like there is a problem with the calculation of this 
                // offset when the sample rate is 2.5 MHz and the filter offset is 0.6 MHz. 
                // Verify if there is a problem here or in the Fold module.
                resamplerOffset = static_cast<U64>(std::round(centerBin));
            }
            
            resamplerSize /= static_cast<U64>(resamplerRatio);
            padSize /= static_cast<U64>(resamplerRatio); 
        }
        
        return true;
    }

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(FilterEngine, is_specialized<Jetstream::Pad<D, IT>>::value &&
                               is_specialized<Jetstream::FFT<D, IT>>::value &&
                               is_specialized<Jetstream::Multiply<D, IT>>::value &&
                               is_specialized<Jetstream::Fold<D, IT>>::value &&
                               is_specialized<Jetstream::FFT<D, IT>>::value &&
                               is_specialized<Jetstream::Unpad<D, IT>>::value &&
                               is_specialized<Jetstream::OverlapAdd<D, IT>>::value &&
                               std::is_same<OT, void>::value)

#endif
