#ifndef JETSTREAM_BLOCKS_BASE_HH
#define JETSTREAM_BLOCKS_BASE_HH

#include "jetstream/types.hh"

#if defined(JETSTREAM_MODULE_FFT_AVAILABLE)
#include "jetstream/blocks/fft.hh"
#define JETSTREAM_BLOCK_FFT_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_LINEPLOT_AVAILABLE)
#include "jetstream/blocks/lineplot.hh"
#define JETSTREAM_BLOCK_LINEPLOT_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_WATERFALL_AVAILABLE)
#include "jetstream/blocks/waterfall.hh"
#define JETSTREAM_BLOCK_WATERFALL_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_SPECTROGRAM_AVAILABLE)
#include "jetstream/blocks/spectrogram.hh"
#define JETSTREAM_BLOCK_SPECTROGRAM_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_CONSTELLATION_AVAILABLE)
#include "jetstream/blocks/constellation.hh"
#define JETSTREAM_BLOCK_CONSTELLATION_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_SOAPY_AVAILABLE)
#include "jetstream/blocks/soapy.hh"
#define JETSTREAM_BLOCK_SOAPY_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_MULTIPLY_AVAILABLE)
#include "jetstream/blocks/multiply.hh"
#define JETSTREAM_BLOCK_MULTIPLY_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_SCALE_AVAILABLE)
#include "jetstream/blocks/scale.hh"
#define JETSTREAM_BLOCK_SCALE_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_PAD_AVAILABLE)
#include "jetstream/blocks/pad.hh"
#define JETSTREAM_BLOCK_PAD_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_UNPAD_AVAILABLE)
#include "jetstream/blocks/unpad.hh"
#define JETSTREAM_BLOCK_UNPAD_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_OVERLAP_ADD_AVAILABLE)
#include "jetstream/blocks/overlap_add.hh"
#define JETSTREAM_BLOCK_OVERLAP_ADD_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_REMOTE_AVAILABLE)
#include "jetstream/blocks/remote.hh"
#define JETSTREAM_BLOCK_REMOTE_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_FILTER_TAPS_AVAILABLE)
#include "jetstream/blocks/filter_taps.hh"
#define JETSTREAM_BLOCK_FILTER_TAPS_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_AMPLITUDE_AVAILABLE)
#include "jetstream/blocks/amplitude.hh"
#define JETSTREAM_BLOCK_AMPLITUDE_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_AGC_AVAILABLE)
#include "jetstream/blocks/agc.hh"
#define JETSTREAM_BLOCK_AGC_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_FM_AVAILABLE)
#include "jetstream/blocks/fm.hh"
#define JETSTREAM_BLOCK_FM_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_AUDIO_AVAILABLE)
#include "jetstream/blocks/audio.hh"
#define JETSTREAM_BLOCK_AUDIO_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_INVERT_AVAILABLE)
#include "jetstream/blocks/invert.hh"
#define JETSTREAM_BLOCK_INVERT_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_WINDOW_AVAILABLE)
#include "jetstream/blocks/window.hh"
#define JETSTREAM_BLOCK_WINDOW_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_MULTIPLY_CONSTANT_AVAILABLE)
#include "jetstream/blocks/multiply_constant.hh"
#define JETSTREAM_BLOCK_MULTIPLY_CONSTANT_AVAILABLE
#endif

#include "jetstream/blocks/expand_dims.hh"
#define JETSTREAM_BLOCK_EXPAND_DIMS_AVAILABLE

#if defined(JETSTREAM_MODULE_PAD_AVAILABLE) && \
    defined(JETSTREAM_MODULE_UNPAD_AVAILABLE) && \
    defined(JETSTREAM_MODULE_OVERLAP_ADD_AVAILABLE) && \
    defined(JETSTREAM_MODULE_MULTIPLY_AVAILABLE) && \
    defined(JETSTREAM_MODULE_FFT_AVAILABLE) && \
    defined(JETSTREAM_MODULE_FOLD_AVAILABLE)
#include "jetstream/blocks/filter_engine.hh"
#define JETSTREAM_BLOCK_FILTER_ENGINE_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_FOLD_AVAILABLE)
#include "jetstream/blocks/fold.hh"
#define JETSTREAM_BLOCK_FOLD_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_CAST_AVAILABLE)
#include "jetstream/blocks/cast.hh"
#define JETSTREAM_BLOCK_CAST_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_SPEECH_RECOGNITION_AVAILABLE)
#include "jetstream/blocks/speech_recognition.hh"
#define JETSTREAM_BLOCK_SPEECH_RECOGNITION_AVAILABLE
#endif

#include "jetstream/blocks/note.hh"
#define JETSTREAM_BLOCK_NOTE_AVAILABLE

#if defined(JETSTREAM_MODULE_TAKE_AVAILABLE)
#include "jetstream/blocks/take.hh"
#define JETSTREAM_BLOCK_TAKE_AVAILABLE
#endif

#include "jetstream/blocks/squeeze_dims.hh"
#define JETSTREAM_BLOCK_SQUEEZE_DIMS_AVAILABLE

#if defined(JETSTREAM_MODULE_SCALE_AVAILABLE) && \
    defined(JETSTREAM_MODULE_AMPLITUDE_AVAILABLE) && \
    defined(JETSTREAM_MODULE_FFT_AVAILABLE) && \
    defined(JETSTREAM_MODULE_AGC_AVAILABLE) && \
    defined(JETSTREAM_MODULE_MULTIPLY_AVAILABLE) && \
    defined(JETSTREAM_MODULE_INVERT_AVAILABLE) && \
    defined(JETSTREAM_MODULE_WINDOW_AVAILABLE)
#include "jetstream/blocks/spectroscope.hh"
#define JETSTREAM_BLOCK_SPECTROSCOPE_AVAILABLE
#endif

#if defined(JETSTREAM_BLOCK_FILTER_TAPS_AVAILABLE) && \
    defined(JETSTREAM_BLOCK_FILTER_ENGINE_AVAILABLE)
#include "jetstream/blocks/filter.hh"
#define JETSTREAM_BLOCK_FILTER_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_DUPLICATE_AVAILABLE)
#include "jetstream/blocks/duplicate.hh"
#define JETSTREAM_BLOCK_DUPLICATE_AVAILABLE
#endif

#if defined(JETSTREAM_MODULE_ARITHMETIC_AVAILABLE)
#include "jetstream/blocks/arithmetic.hh"
#define JETSTREAM_BLOCK_ARITHMETIC_AVAILABLE
#endif

#include "jetstream/blocks/slice.hh"
#define JETSTREAM_BLOCK_SLICE_AVAILABLE

// [NEW BLOCK HOOK]

#endif  // JETSTREAM_BLOCKS_BASE_HH
