#ifndef JETSTREAM_MODULES_BASE_HH
#define JETSTREAM_MODULES_BASE_HH

#include "jetstream/types.hh"

#ifdef JETSTREAM_MODULE_FFT_AVAILABLE
#include "jetstream/modules/fft.hh"
#endif

#ifdef JETSTREAM_MODULE_FILTER_TAPS_AVAILABLE
#include "jetstream/modules/filter_taps.hh"
#endif

#ifdef JETSTREAM_MODULE_WINDOW_AVAILABLE
#include "jetstream/modules/window.hh"
#endif

#ifdef JETSTREAM_MODULE_MULTIPLY_AVAILABLE
#include "jetstream/modules/multiply.hh"
#endif

#ifdef JETSTREAM_MODULE_AMPLITUDE_AVAILABLE
#include "jetstream/modules/amplitude.hh"
#endif

#ifdef JETSTREAM_MODULE_SCALE_AVAILABLE
#include "jetstream/modules/scale.hh"
#endif

#ifdef JETSTREAM_MODULE_SOAPY_AVAILABLE
#include "jetstream/modules/soapy.hh"
#endif

#ifdef JETSTREAM_MODULE_AUDIO_AVAILABLE
#include "jetstream/modules/audio.hh"
#endif

#ifdef JETSTREAM_MODULE_FM_AVAILABLE
#include "jetstream/modules/fm.hh"
#endif

#ifdef JETSTREAM_MODULE_INVERT_AVAILABLE
#include "jetstream/modules/invert.hh"
#endif

#ifdef JETSTREAM_MODULE_MULTIPLY_CONSTANT_AVAILABLE
#include "jetstream/modules/multiply_constant.hh"
#endif

#ifdef JETSTREAM_MODULE_CAST_AVAILABLE
#include "jetstream/modules/cast.hh"
#endif

#ifdef JETSTREAM_MODULE_PAD_AVAILABLE
#include "jetstream/modules/pad.hh"
#endif

#ifdef JETSTREAM_MODULE_UNPAD_AVAILABLE
#include "jetstream/modules/unpad.hh"
#endif

#ifdef JETSTREAM_MODULE_OVERLAP_ADD_AVAILABLE
#include "jetstream/modules/overlap_add.hh"
#endif

#ifdef JETSTREAM_MODULE_AGC_AVAILABLE
#include "jetstream/modules/agc.hh"
#endif

#ifdef JETSTREAM_MODULE_LINEPLOT_AVAILABLE
#include "jetstream/modules/lineplot.hh"
#endif

#ifdef JETSTREAM_MODULE_WATERFALL_AVAILABLE
#include "jetstream/modules/waterfall.hh"
#endif

#ifdef JETSTREAM_MODULE_SPECTROGRAM_AVAILABLE
#include "jetstream/modules/spectrogram.hh"
#endif

#ifdef JETSTREAM_MODULE_CONSTELLATION_AVAILABLE
#include "jetstream/modules/constellation.hh"
#endif

#include "jetstream/modules/tensor_modifier.hh"

#ifdef JETSTREAM_MODULE_FOLD_AVAILABLE
#include "jetstream/modules/fold.hh"
#endif

#ifdef JETSTREAM_MODULE_TAKE_AVAILABLE
#include "jetstream/modules/take.hh"
#endif

#ifdef JETSTREAM_MODULE_DUPLICATE_AVAILABLE
#include "jetstream/modules/duplicate.hh"
#endif

#ifdef JETSTREAM_MODULE_FILE_WRITER_AVAILABLE
#include "jetstream/modules/file_writer.hh"
#endif

#ifdef JETSTREAM_MODULE_FILE_READER_AVAILABLE
#include "jetstream/modules/file_reader.hh"
#endif

#ifdef JETSTREAM_MODULE_THROTTLE_AVAILABLE
#include "jetstream/modules/throttle.hh"
#endif

// [NEW MODULE HOOK]

#endif  // JETSTREAM_MODULES_BASE_HH
