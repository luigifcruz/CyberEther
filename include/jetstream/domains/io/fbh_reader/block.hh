#ifndef JETSTREAM_DOMAINS_IO_FBH_READER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_FBH_READER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FbhReader : public Block::Config {
    std::string filepath  = "";
    U64         batchSize = 256;
    bool        loop      = true;
    bool        playing   = true;

    JST_BLOCK_TYPE(fbh_reader);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(filepath, batchSize, loop, playing);
    JST_BLOCK_DESCRIPTION(
        "FBH5 Reader",
        "Reads beamformer filterbank data from an FBH5 file.",
        "# FBH5 Reader\n"
        "Reads channelized filterbank data from an FBH5 (HDF5) archive and emits "
        "it as float32 spectrogram chunks. FBH5 files store data in the dataset "
        "`/data` with shape `[ntimes, nifs, nchans]`.\n\n"

        "## Arguments\n"
        "- **File Path**: Path to the `.fbh5` or `.h5` file.\n"
        "- **Batch Size**: Number of time rows to read per processing cycle.\n"
        "- **Loop**: Restart from the beginning when the end of file is reached.\n"
        "- **Playing**: Pause or resume reading.\n\n"

        "## Outputs\n"
        "- **Signal**: F32 tensor `[batchSize, nifs × nchans]` — one row per time sample.\n\n"

        "## Notes\n"
        "Uses the POSIX HDF5 VFD (no GPUDirect Storage required). "
        "Compatible with any system that has libhdf5 installed."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_FBH_READER_BLOCK_HH
