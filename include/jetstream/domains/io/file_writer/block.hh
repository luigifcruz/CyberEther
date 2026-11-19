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
        "Writes each incoming buffer to a Raw I/Q, SigMF, or WAV file while Recording "
        "is on. It usually follows a radio source, to capture samples that a File "
        "Reader plays back later.\n"
        "\n"
        "- **Raw I/Q files have no header.** Type and sample rate are lost, so use "
        "SigMF or WAV to keep them.\n"
        "- **Retunes are saved in SigMF only.** Each change of the input `frequency` "
        "starts a new capture, and WAV keeps the first.\n"
        "- **Recording never appends.** Restarting it, or changing a setting while it "
        "runs, needs Overwrite on and rewrites the file.\n"
        "- **Lost samples are marked in SigMF.** With an SDR source, the .sigmf-meta "
        "file counts them and annotates the buffer where each shows up.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32, F32, CI16, I16, CU16, U16, CI8, I8, CU8, or U8 | Any "
        "shape in a contiguous buffer, written in memory order with complex values as "
        "real then imaginary parts. A WAV file rejects CI8, I8, CU16, and U16. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **File Path** | Empty | A path in an existing folder | File to write. For "
        "SigMF, it writes a .sigmf-data and a .sigmf-meta file with the same base name."
        " |\n"
        "| **File Format** | Raw I/Q | Raw I/Q, SigMF, WAV | Raw I/Q writes headerless "
        "samples. A SigMF recording also saves the type, `sampleRate`, and `frequency` "
        "in the .sigmf-meta file. A WAV file saves the type and `sampleRate` in its "
        "header, and the starting `frequency` in an auxi chunk. |\n"
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
        "| Capture a radio as WAV | File Format WAV | 8,192 complex samples at 2 MS/s |"
        " 64 KiB per buffer after the header, as two float channels at 2 MS/s |\n"
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
        "| The log warns that the file path is empty or its folder is missing | "
        "Recording is on with no File Path, or its folder does not exist | Pick a File "
        "Path in an existing folder. |\n"
        "| A .wav file does not play, or plays at 0 Hz | File Format is Raw I/Q, or the"
        " input has no `sampleRate` | Set File Format to WAV, and record from a source "
        "that sets `sampleRate`. |\n"
        "| The .sigmf-meta file shows a nonzero `cyberether:buffer_lost_samples`, "
        "`cyberether:radio_lost_samples`, or `cyberether:radio_gaps` | The disk or the "
        "chain fell behind the source, or the radio was not read in time | Record CI16 "
        "through a **Cast**, raise Buffer Multiplier in the source, or record to a "
        "faster disk. |\n"
        "| The block reports that a contiguous tensor is expected | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer is written and flushed during the compute step, so a slow disk "
        "slows the whole chain. The SigMF metadata is rewritten at the start, on each "
        "new capture or loss, and when recording stops. The WAV header is finished with"
        " the final size when recording stops, in the large-file variant when the data "
        "outgrows a plain header.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **File Reader** plays the file back with a matching type.\n"
        "- Radio samples to record come from **SDR**."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_FILE_WRITER_BLOCK_HH
