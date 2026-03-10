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
    JST_BLOCK_PARAMS(filepath, fileFormat, overwrite, recording);
    JST_BLOCK_DESCRIPTION(
        "File Writer",
        "Writes raw binary signal data to a file.",
        "# File Writer\n"
        "The File Writer block writes incoming signal data to a raw binary file. "
        "It supports various data types and can optionally overwrite existing files.\n\n"

        "## Arguments\n"
        "- **File Path**: Path to the output file.\n"
        "- **File Format**: The format of the output file (e.g., Raw).\n"
        "- **Overwrite**: Whether to overwrite the file if it already exists.\n"
        "- **Recording**: Start or stop recording to the file.\n\n"

        "## Useful For\n"
        "- Recording signal data for later playback or analysis.\n"
        "- Capturing raw IQ samples from SDR sources.\n"
        "- Saving processed signal data to disk.\n\n"

        "## Examples\n"
        "- Record IQ samples to file:\n"
        "  Config: File Path='capture.raw', Overwrite=true, Recording=true\n"
        "  Input: CF32[8192] -> Written to file each cycle.\n"
        "- Record U16 samples to file:\n"
        "  Config: File Path='capture_u16.raw', Overwrite=true, Recording=true\n"
        "  Input: U16[8192] -> Written to file each cycle.\n\n"

        "## Implementation\n"
        "Input Buffer -> FileWriter Module\n"
        "1. Opens the specified file in binary write mode when recording is enabled.\n"
        "2. Writes incoming samples to the file each compute cycle.\n"
        "3. Closes the file when recording is stopped."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_FILE_WRITER_BLOCK_HH
