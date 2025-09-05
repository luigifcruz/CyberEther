#ifndef JETSTREAM_BLOCKS_MANIFEST_HH
#define JETSTREAM_BLOCKS_MANIFEST_HH

#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "jetstream/blocks/base.hh"

namespace Jetstream::Blocks {

#define JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, INPUT, OUTPUT) \
    Jetstream::Blocks::AddBlockToConstructorManifest<BLOCK, DEVICE, INPUT, OUTPUT>(constructorManifest); \
    Jetstream::Blocks::AddBlockToMetadataManifest<BLOCK, DEVICE, INPUT, OUTPUT>(metadataManifest);

#define JST_BLOCKS_MANIFEST_TYPE(BLOCK, DEVICE) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, void, void) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, CF32, void) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, F32, void) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, void, CF32) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, void, F32) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, CF32, CF32) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, CF32, F32) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, F32, CF32) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, F32, F32) \
    JST_BLOCKS_MANIFEST_ADD(BLOCK, DEVICE, F32, I16)
    
#define JST_BLOCKS_MANIFEST_DEVICE(BLOCK) \
    JST_BLOCKS_MANIFEST_TYPE(BLOCK, Device::CPU) \
    JST_BLOCKS_MANIFEST_TYPE(BLOCK, Device::Metal) \
    JST_BLOCKS_MANIFEST_TYPE(BLOCK, Device::CUDA) \
    JST_BLOCKS_MANIFEST_TYPE(BLOCK, Device::Vulkan)

#define JST_BLOCKS_MANIFEST(...) \
    FOR_EACH(JST_BLOCKS_MANIFEST_DEVICE, __VA_ARGS__)

template<template<Device, typename...> class B, Device D, typename IT, typename OT>
inline void AddBlockToConstructorManifest(Block::ConstructorManifest& manifest) {
    using BlockType = B<D, IT, OT>;
    if constexpr (is_specialized<BlockType>::value && 
                  Backend::GetBackend<D>::enabled) {
        BlockType block;
        const Block::Fingerprint fingerprint = {
            .id = block.id(),
            .device = GetDeviceName(block.device()), 
            .inputDataType = NumericTypeInfo<IT>::name,
            .outputDataType = NumericTypeInfo<OT>::name
        };

        if (manifest.contains(fingerprint)) {
            JST_WARN("[BLOCKS] Skipping duplicate block constructor: '{}'", fingerprint);
            return;
        }
        JST_TRACE("[BLOCKS] Adding block constructor to manifest: '{}'", fingerprint);

        manifest[fingerprint] = [](Instance& instance, 
                                   const std::string& id, 
                                   Parser::RecordMap& i, 
                                   Parser::RecordMap& c, 
                                   Parser::RecordMap& s) {
            return instance.addBlock<B, D, IT, OT>(id, i, c, s);
        };
    }
}

template<template<Device, typename...> class B, Device D, typename IT, typename OT>
inline void AddBlockToMetadataManifest(Block::MetadataManifest& manifest) {
    using BlockType = B<D, IT, OT>;
    if constexpr (is_specialized<BlockType>::value && 
                  Backend::GetBackend<D>::enabled) {
        BlockType block;

        // Add block metadata to manifest.
        if (!manifest.contains(block.id())) {
            manifest[block.id()] = {
                .title = block.name(),
                .summary = block.summary(),
                .description = block.description(),
                .options = {}
            };
        }
        
        // Add block options to manifest.
        manifest[block.id()].options[block.device()].push_back({
            NumericTypeInfo<IT>::name,
            NumericTypeInfo<OT>::name
        });
    }
}

inline void GetDefaultManifest(Block::ConstructorManifest& constructorManifest, 
                               Block::MetadataManifest& metadataManifest) {
    JST_TRACE("[BLOCKS] Getting default block manifest list.");

#ifdef JETSTREAM_BLOCK_FFT_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::FFT);
#endif
#ifdef JETSTREAM_BLOCK_LINEPLOT_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Lineplot);
#endif
#ifdef JETSTREAM_BLOCK_WATERFALL_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Waterfall);
#endif
#ifdef JETSTREAM_BLOCK_SPECTROGRAM_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Spectrogram);
#endif
#ifdef JETSTREAM_BLOCK_CONSTELLATION_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Constellation);
#endif
#ifdef JETSTREAM_BLOCK_SOAPY_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Soapy);
#endif
#ifdef JETSTREAM_BLOCK_MULTIPLY_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Multiply);
#endif
#ifdef JETSTREAM_BLOCK_SCALE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Scale);
#endif
#ifdef JETSTREAM_BLOCK_PAD_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Pad);
#endif
#ifdef JETSTREAM_BLOCK_UNPAD_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Unpad);
#endif
#ifdef JETSTREAM_BLOCK_OVERLAP_ADD_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::OverlapAdd);
#endif
#ifdef JETSTREAM_BLOCK_REMOTE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Remote);
#endif
#ifdef JETSTREAM_BLOCK_FILTER_TAPS_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::FilterTaps);
#endif
#ifdef JETSTREAM_BLOCK_AMPLITUDE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Amplitude);
#endif
#ifdef JETSTREAM_BLOCK_AGC_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::AGC);
#endif
#ifdef JETSTREAM_BLOCK_FM_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::FM);
#endif
#ifdef JETSTREAM_BLOCK_AUDIO_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Audio);
#endif
#ifdef JETSTREAM_BLOCK_INVERT_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Invert);
#endif
#ifdef JETSTREAM_BLOCK_WINDOW_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Window);
#endif
#ifdef JETSTREAM_BLOCK_MULTIPLY_CONSTANT_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::MultiplyConstant);
#endif
#ifdef JETSTREAM_BLOCK_EXPAND_DIMS_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::ExpandDims);
#endif
#ifdef JETSTREAM_BLOCK_FILTER_ENGINE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::FilterEngine);
#endif
#ifdef JETSTREAM_BLOCK_FOLD_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Fold);
#endif
#ifdef JETSTREAM_BLOCK_CAST_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Cast);
#endif
#ifdef JETSTREAM_BLOCK_SPEECH_RECOGNITION_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::SpeechRecognition);
#endif
#ifdef JETSTREAM_BLOCK_NOTE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Note);
#endif

#ifdef JETSTREAM_BLOCK_SQUEEZE_DIMS_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::SqueezeDims);
#endif
#ifdef JETSTREAM_BLOCK_SPECTROSCOPE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Spectroscope);
#endif
#ifdef JETSTREAM_BLOCK_FILTER_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Filter);
#endif
#ifdef JETSTREAM_BLOCK_DUPLICATE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Duplicate);
#endif
#ifdef JETSTREAM_BLOCK_ARITHMETIC_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Arithmetic);
#endif
#ifdef JETSTREAM_BLOCK_SLICE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Slice);
#endif
#ifdef JETSTREAM_BLOCK_RESHAPE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Reshape);
#endif
#ifdef JETSTREAM_BLOCK_FILE_WRITER_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::FileWriter);
#endif
#ifdef JETSTREAM_BLOCK_FILE_READER_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::FileReader);
#endif
#ifdef JETSTREAM_BLOCK_THROTTLE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Throttle);
#endif
#ifdef JETSTREAM_BLOCK_DECIMATOR_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::Decimator);
#endif
#ifdef JETSTREAM_BLOCK_SPECTRUM_ENGINE_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::SpectrumEngine);
#endif
#ifdef JETSTREAM_BLOCK_SIGNAL_GENERATOR_AVAILABLE
    JST_BLOCKS_MANIFEST(Blocks::SignalGenerator);
#endif
    // [NEW BLOCK HOOK]
}

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_BLOCKS_MANIFEST_HH