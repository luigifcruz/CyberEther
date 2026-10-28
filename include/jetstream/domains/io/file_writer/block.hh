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
        "Writes signal samples to a file.",
        "# File Writer\n"
        "\n"
        "Writes each incoming buffer to a raw or SigMF file while Recording is on. It "
        "usually follows a radio source, to capture samples that a File Reader plays "
        "back later.\n"
        "\n"
        "- **Raw files have no header.** Type and sample rate are lost, so use SigMF to"
        " keep them.\n"
        "- **SigMF tracks retunes.** Each change of the input `frequency` starts a new "
        "capture in the metadata.\n"
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
        "| **File Path** | Empty | A path in an existing folder | File to write. SigMF "
        "writes a .sigmf-data and a .sigmf-meta file with the same base name. |\n"
        "| **File Format** | Raw | Raw, SigMF | Raw writes headerless samples. SigMF "
        "also saves the type, `sampleRate`, and `frequency` in the .sigmf-meta file. |\n"
        "| **Overwrite** | Off | On, Off | Lets a recording replace a file that already"
        " exists. |\n"
        "| **Recording** | Off | On, Off | Writes every buffer while on. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **File Size** | Size of the data file at File Path, counting up while "
        "recording, in B, KiB, MiB, or GiB. |\n"
        "| **Throughput** | Smoothed write rate in MB/s. Reads 0 when not recording. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume File Path is set and Recording is on.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Capture radio samples | Defaults | 8,192 complex samples at 2 MS/s | 64 KiB "
        "per buffer, with Throughput near 16 MB/s |\n"
        "| Record demodulated audio | Defaults | 819 real samples at 200 kS/s | 3,276 "
        "bytes per buffer, with Throughput near 0.8 MB/s |\n"
        "| Capture a radio as SigMF | File Format SigMF | 8,192 complex samples at 2 "
        "MS/s | 64 KiB per buffer, and a .sigmf-meta file with cf32_le at 2 MS/s |\n"
        "| Replace an earlier take | Overwrite on | 8,192 complex samples | The file "
        "restarts empty and grows by 64 KiB per buffer |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the file already exists | A file sits at File Path, "
        "or a SigMF file with the same base name, and Overwrite is off | Turn on "
        "Overwrite, or pick a new File Path. |\n"
        "| The log warns that the file path is empty | Recording is on with no File "
        "Path | Pick a File Path. |\n"
        "| The log warns that the parent directory does not exist | The folder in File "
        "Path is missing | Create the folder, or pick another path. |\n"
        "| A .wav file does not play | Raw writes samples with no header | Read it as "
        "raw data, such as with a File Reader set to the same type. |\n"
        "| The block reports that a contiguous tensor is expected | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer is written and flushed during the compute step, so a slow disk "
        "slows the whole chain. The SigMF metadata is rewritten at the start, on each "
        "new capture, and when recording stops.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **File Reader** plays the file back with a matching type.\n"
        "- Radio samples to record come from **SDR**."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_FILE_WRITER_BLOCK_HH
