#ifndef JETSTREAM_DOMAINS_IO_FILE_READER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_FILE_READER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FileReader : public Block::Config {
    std::string filepath = "";
    std::string fileFormat = "raw";
    std::string dataType = "CF32";
    U64 batchSize = 8192;
    bool loop = true;
    bool playing = true;

    JST_BLOCK_TYPE(file_reader);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(filepath, fileFormat, dataType, batchSize, loop, playing);
    JST_BLOCK_DESCRIPTION(
        "File Reader",
        "Reads raw binary signal data from a file.",
        "# File Reader\n"
        "\n"
        "Reads raw samples from a file, one buffer per cycle, looping back at the end "
        "when asked. It usually starts a chain in place of a radio, feeding a Throttle "
        "and then a Spectrum Analyzer.\n"
        "\n"
        "- **Playback is not paced.** Each cycle reads the next buffer as fast as the "
        "chain runs.\n"
        "- **The output has no sample rate.** Neither `sampleRate` nor `frequency` is "
        "set, so views show no MHz axis.\n"
        "- **Samples come out as stored.** Integer types are not scaled, so convert "
        "them with a Cast block.\n"
        "- **Every file is read as raw samples.** A .wav header comes out as samples "
        "too.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | Data Type, CF32 by default | Batch Size samples on a "
        "`sampleAxis`, with no other axis. Neither `sampleRate` nor `frequency` is set."
        " |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **File Path** | Empty | An existing file | File to read. The file dialog "
        "offers .bin, .raw, .iq, .wav, and .dat. |\n"
        "| **File Format** | Raw | Raw | Headerless samples. |\n"
        "| **Data Type** | CF32 | CF64, F64, CF32, F32, CI8, I8, CU8, U8, CI16, I16, "
        "CU16, U16 | Type of each stored sample, and of the output. It must match how "
        "the file was written. |\n"
        "| **Batch Size** | 8,192 samples | 1 or more | Samples per output buffer, and "
        "the FFT size downstream. |\n"
        "| **Loop** | On | On, Off | Restarts from the beginning at the end of the "
        "file. Off repeats the last buffer from then on. |\n"
        "| **Playing** | On | On, Off | Reads a new buffer every cycle. Off repeats the"
        " current buffer. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Position** | Share of the file read in the current pass, from 0 to 100%. |\n"
        "| **Bandwidth** | Smoothed read rate in MB/s, with a MB of 1,048,576 bytes. "
        "Keeps its last value while paused. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Play a radio capture | Defaults | 8,192 complex samples per buffer, 64 KB of"
        " the file each |\n"
        "| Play an 8-bit capture | Data Type CU8 | 8,192 complex samples per buffer "
        "from 0 to 255, 16 KB of the file each |\n"
        "| Play a 16-bit capture | Data Type CI16, Batch Size 4,096 | 4,096 complex "
        "samples per buffer, 16 KB of the file each |\n"
        "| Play a real recording | Data Type F32, Batch Size 4,800 | 4,800 real samples"
        " per buffer, 18.75 KB of the file each |\n"
        "| Play once | Loop off | 8,192 complex samples per buffer until the end, then "
        "the last buffer again |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the file path is empty | File Path is not set, the "
        "default | Pick a file in File Path. |\n"
        "| The block reports that the file does not exist | The file at File Path was "
        "moved or deleted | Pick the file again. |\n"
        "| The spectrum looks like noise | Data Type does not match how the file was "
        "written | Set Data Type to the type the recorder used, such as the input type "
        "of a File Writer. |\n"
        "| A .wav file starts each pass with a burst of noise | The header is read as "
        "samples, which also shifts every sample when the header is not a whole number "
        "of samples | Convert the file to raw samples first. |\n"
        "| Views keep showing the same buffer | Playing is off, or the file ended with "
        "Loop off | Turn on Playing, or turn on Loop. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each cycle reads the next buffer from the file during the compute step, so a "
        "slow disk slows the whole chain. A short read at the end of the file leaves "
        "the rest of the buffer holding older samples.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **File Writer** records files that this block plays.\n"
        "- **Throttle** paces playback to real time.\n"
        "- **Cast** converts integer samples to CF32.\n"
        "- **Spectrum Analyzer** shows the recorded band."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_FILE_READER_BLOCK_HH
