#ifndef JETSTREAM_DOMAINS_IO_FILE_WRITER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_FILE_WRITER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FileWriter : public Block::Config {
    std::string filepath = "";
    std::string fileFormat = "raw";
    bool overwrite = false;
    bool recording = false;

    JST_BLOCK_TYPE(file_writer);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(filepath, fileFormat, overwrite, recording);
    JST_BLOCK_DESCRIPTION(
        "File Writer",
        "Writes raw binary signal data to a file.",
        "# File Writer\n"
        "\n"
        "Writes each incoming buffer to a file as raw bytes while Recording is on. It "
        "usually follows a radio source, to capture samples that a File Reader plays "
        "back later.\n"
        "\n"
        "- **The file has no header.** Type, shape, and sample rate are not saved, so "
        "note them for playback.\n"
        "- **Recording never appends.** Restarting it, or changing a setting while it "
        "runs, needs Overwrite on and rewrites the file.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32, F32, CI16, I16, CU16, U16, CI8, I8, CU8, or U8 | Any "
        "shape, written in memory order. Complex values are stored as real then "
        "imaginary parts. The buffer must be contiguous. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **File Path** | Empty | A path in an existing folder | File to write. The "
        "save dialog offers .bin, .raw, .iq, .wav, and .dat. |\n"
        "| **File Format** | Raw | Raw | Headerless samples. |\n"
        "| **Overwrite** | Off | On, Off | Lets a recording replace a file that already"
        " exists. |\n"
        "| **Recording** | Off | On, Off | Writes every buffer while on. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **File Size** | Size of the file at File Path, counting up while recording, "
        "in B, KB, MB, or GB. Each unit is 1,024 of the previous one. |\n"
        "| **Bandwidth** | Smoothed write rate in MB/s, with a MB of 1,048,576 bytes. "
        "Reads 0 when not recording. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume File Path is set and Recording is on.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Capture radio samples | Defaults | 8,192 complex samples at 2 MS/s | 64 KB "
        "per buffer, with Bandwidth near 15.3 MB/s |\n"
        "| Record demodulated audio | Defaults | 819 real samples at 200 kS/s | 3,276 "
        "bytes per buffer, with Bandwidth near 0.8 MB/s |\n"
        "| Batched capture | Defaults | 8 batches x 8,192 complex samples | 512 KB per "
        "buffer, batches back to back |\n"
        "| Replace an earlier take | Overwrite on | 8,192 complex samples | The file "
        "restarts empty and grows by 64 KB per buffer |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the file already exists | A file sits at File Path, "
        "such as an earlier take, and Overwrite is off | Turn on Overwrite, or pick a "
        "new File Path. |\n"
        "| The log warns that the file path is empty | Recording is on with no File "
        "Path | Pick a File Path. |\n"
        "| The log warns that the parent directory does not exist | The folder in File "
        "Path is missing | Create the folder, or pick another path. |\n"
        "| A .wav file does not play | The block writes raw samples with no header | "
        "Read it as raw data, such as with a File Reader set to the same type. |\n"
        "| The block reports that a contiguous tensor is expected | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer is written and flushed during the compute step, so a slow disk "
        "slows the whole chain.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **File Reader** plays the file back with a matching type.\n"
        "- **Soapy SDR** provides radio samples to record."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_FILE_WRITER_BLOCK_HH
