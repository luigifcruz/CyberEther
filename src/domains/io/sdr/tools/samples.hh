#ifndef JETSTREAM_DOMAINS_IO_SDR_TOOLS_SAMPLES_HH
#define JETSTREAM_DOMAINS_IO_SDR_TOOLS_SAMPLES_HH

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

JETSTREAM_API void ConvertU8(const U8* input, CF32* output, U64 count);
JETSTREAM_API void ConvertI8(const I8* input, CF32* output, U64 count);
JETSTREAM_API void ConvertI16(const I16* input, CF32* output, U64 count, F32 scale);

JETSTREAM_API U64 GuardSamples(F64 seconds, F64 sampleRate);

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_TOOLS_SAMPLES_HH
