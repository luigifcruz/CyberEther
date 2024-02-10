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
    
    JST_BLOCKS_MANIFEST(
#ifdef JETSTREAM_BLOCK_FFT_AVAILABLE
        Blocks::FFT,
#endif
#ifdef JETSTREAM_BLOCK_LINEPLOT_AVAILABLE
        Blocks::Lineplot,
#endif
#ifdef JETSTREAM_BLOCK_WATERFALL_AVAILABLE
        Blocks::Waterfall,
#endif
#ifdef JETSTREAM_BLOCK_SPECTROGRAM_AVAILABLE
        Blocks::Spectrogram,
#endif
#ifdef JETSTREAM_BLOCK_CONSTELLATION_AVAILABLE
        Blocks::Constellation,
#endif
#ifdef JETSTREAM_BLOCK_SOAPY_AVAILABLE
        Blocks::Soapy,
#endif
#ifdef JETSTREAM_BLOCK_MULTIPLY_AVAILABLE
        Blocks::Multiply,
#endif
#ifdef JETSTREAM_BLOCK_SCALE_AVAILABLE
        Blocks::Scale,
#endif
#ifdef JETSTREAM_BLOCK_PAD_AVAILABLE
        Blocks::Pad,
#endif
#ifdef JETSTREAM_BLOCK_UNPAD_AVAILABLE
        Blocks::Unpad,
#endif
#ifdef JETSTREAM_BLOCK_OVERLAP_ADD_AVAILABLE
        Blocks::OverlapAdd,
#endif
#ifdef JETSTREAM_BLOCK_REMOTE_AVAILABLE
        Blocks::Remote,
#endif
#ifdef JETSTREAM_BLOCK_FILTER_TAPS_AVAILABLE
        Blocks::FilterTaps,
#endif
#ifdef JETSTREAM_BLOCK_AMPLITUDE_AVAILABLE
        Blocks::Amplitude,
#endif
#ifdef JETSTREAM_BLOCK_AGC_AVAILABLE
        Blocks::AGC,
#endif
#ifdef JETSTREAM_BLOCK_FM_AVAILABLE
        Blocks::FM,
#endif
#ifdef JETSTREAM_BLOCK_AUDIO_AVAILABLE
        Blocks::Audio,
#endif
#ifdef JETSTREAM_BLOCK_INVERT_AVAILABLE
        Blocks::Invert,
#endif
#ifdef JETSTREAM_BLOCK_WINDOW_AVAILABLE
        Blocks::Window,
#endif
#ifdef JETSTREAM_BLOCK_MULTIPLY_CONSTANT_AVAILABLE
        Blocks::MultiplyConstant,
#endif
#ifdef JETSTREAM_BLOCK_EXPAND_DIMS_AVAILABLE
        Blocks::ExpandDims,
#endif
#ifdef JETSTREAM_BLOCK_FILTER_ENGINE_AVAILABLE
        Blocks::FilterEngine,
#endif
#ifdef JETSTREAM_BLOCK_FOLD_AVAILABLE
        Blocks::Fold,
#endif
#ifdef JETSTREAM_BLOCK_CAST_AVAILABLE
        Blocks::Cast,
#endif
#ifdef JETSTREAM_BLOCK_SPEECH_RECOGNITION_AVAILABLE
        Blocks::SpeechRecognition,
#endif
#ifdef JETSTREAM_BLOCK_NOTE_AVAILABLE
        Blocks::Note,
#endif
#ifdef JETSTREAM_BLOCK_TAKE_AVAILABLE
        Blocks::Take,
#endif
#ifdef JETSTREAM_BLOCK_SQUEEZE_DIMS_AVAILABLE
        Blocks::SqueezeDims,
#endif
#ifdef JETSTREAM_BLOCK_SPECTROSCOPE_AVAILABLE
        Blocks::Spectroscope,
#endif
#ifdef JETSTREAM_BLOCK_FILTER_AVAILABLE
        Blocks::Filter,
#endif
#ifdef JETSTREAM_BLOCK_DUPLICATE_AVAILABLE
        Blocks::Duplicate,
#endif
        // [NEW BLOCK HOOK]
    )
}

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_BLOCKS_MANIFEST_HH