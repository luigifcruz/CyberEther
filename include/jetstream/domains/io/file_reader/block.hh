#ifndef JETSTREAM_DOMAINS_IO_FILE_READER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_FILE_READER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FileReader : public Block::Config {
    std::string filepath = "";
    std::string fileFormat = "auto";
    std::string dataType = "CF32";
    U64 batchSize = 8192;
    bool loop = true;
    bool playing = true;

    JST_BLOCK_TYPE(file_reader);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(filepath, fileFormat, dataType, batchSize, loop, playing);
    JST_BLOCK_DESCRIPTION(
        "File Reader",
        "Reads signal samples from a file.",
        "# File Reader\n"
        "\n"
        "Reads samples from a Raw I/Q, SigMF, or WAV file, one buffer per cycle, "
        "looping back at the end when asked. It usually starts a chain in place of a "
        "radio, feeding a Throttle and then a Spectrum Analyzer.\n"
        "\n"
        "- **Playback is not paced.** Add a Throttle with Pacing set to Sample Rate to "
        "play in real time.\n"
        "- **Raw I/Q files carry no metadata.** They set neither `sampleRate` nor "
        "`frequency`, so views show no MHz axis.\n"
        "- **Samples come out as stored.** Integer types are not scaled, so convert "
        "them with a Cast block.\n"
        "- **Two WAV channels become complex samples.** The left channel is I and the "
        "right channel is Q.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | Data Type, or the recorded type for SigMF and WAV. | "
        "Samples on a `sampleAxis`, with no other axis. Recordings in SigMF and WAV set"
        " `sampleRate`, and `frequency` from the current capture or the auxi chunk. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **File Path** | Empty | An existing file or stream | File to read, or a "
        "stream such as `/dev/stdin` or `/dev/zero`. For SigMF, pick the .sigmf-meta or"
        " .sigmf-data file. A WAV file cannot be a stream. |\n"
        "| **File Format** | Auto | Auto, Raw I/Q, SigMF, WAV | Auto picks SigMF or WAV"
        " by the file extension, and Raw I/Q otherwise. Raw I/Q reads headerless "
        "samples. A SigMF or WAV recording gives its type, sample rate, and frequency. "
        "|\n"
        "| **Data Type** | CF32 | CF64, F64, CF32, F32, CI8, I8, CU8, U8, CI16, I16, "
        "CU16, U16 | Type of each stored sample, and of the output. It must match how "
        "the file was written. Shown only when the file is read as Raw I/Q. |\n"
        "| **Samples** | 8,192 samples | 1 or more | Samples per output buffer, and the"
        " FFT size downstream. |\n"
        "| **Loop** | On | On, Off | Restarts from the beginning at the end of the "
        "file. Off repeats the last buffer from then on. Streams ignore it. |\n"
        "| **Playing** | On | On, Off | Reads a new buffer every cycle. Off repeats the"
        " current buffer and stops reading a stream. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Position** | Share of the file read in the current pass, from 0 to 100%. "
        "Data read so far from a stream, in B, KiB, MiB, or GiB. |\n"
        "| **Throughput** | Smoothed read rate in MB/s. Keeps its last value while "
        "paused. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Play a radio capture | Defaults | 8,192 complex samples per buffer, 64 KiB "
        "of the file each |\n"
        "| Play an 8-bit capture | Data Type CU8 | 8,192 complex samples per buffer "
        "from 0 to 255, 16 KiB of the file each |\n"
        "| Play a SigMF or WAV recording | Defaults | 8,192 samples per buffer of the "
        "recorded type, with the recorded sample rate and frequency |\n"
        "| Play samples piped in | File Path `/dev/stdin` | 8,192 complex samples per "
        "buffer as the data arrives |\n"
        "| Play once | Loop off | 8,192 complex samples per buffer until the end, then "
        "the last buffer again |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the file path is empty or the file does not exist | "
        "File Path is not set, or the file was moved or deleted | Pick the file in File"
        " Path. |\n"
        "| The block reports that a SigMF or WAV recording is not supported | The "
        "recording is big-endian, has 24 or 32-bit integers or too many channels, is a "
        ".sigmf archive, or is a WAV stream | Extract the archive, convert the "
        "recording to a supported type, or save the stream to a file. |\n"
        "| The spectrum looks like noise | Data Type does not match how the file was "
        "written | Set Data Type to the type the recorder used, such as the input type "
        "of a File Writer. |\n"
        "| A .wav file starts each pass with a burst of noise | File Format is Raw I/Q,"
        " so the header is read as samples | Set File Format to WAV. |\n"
        "| Views keep showing the same buffer | Playing is off, the file ended with "
        "Loop off, or a stream ended or has no new data | Turn on Playing or Loop, or "
        "feed the stream more data. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each cycle reads the next buffer from the file during the compute step, so a "
        "slow disk slows the whole chain. A short read at the end of the file leaves "
        "the rest of the buffer holding older samples. Streams are read without "
        "blocking while a separate thread waits for new data, and their last buffer is "
        "padded with zeros.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **File Writer** records files that this block plays.\n"
        "- **Throttle** paces playback to the recorded sample rate.\n"
        "- **Cast** scales integer samples to F32 or CF32.\n"
        "- **Spectrum Analyzer** shows the recorded band."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_FILE_READER_BLOCK_HH
