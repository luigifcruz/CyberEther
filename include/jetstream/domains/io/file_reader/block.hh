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
        "The File Reader block reads raw binary signal data from a file and outputs it as a tensor. "
        "It supports various data types and can optionally loop back to the beginning when reaching "
        "the end of the file.\n\n"

        "## Arguments\n"
        "- **File Path**: Path to the raw binary file to read.\n"
        "- **File Format**: The format of the input file (e.g., Raw).\n"
        "- **Data Type**: The data type of samples in the file.\n"
        "- **Batch Size**: Number of samples to read per processing cycle.\n"
        "- **Loop**: Whether to loop back to the start when reaching the end of the file.\n"
        "- **Playing**: Start or stop reading from the file.\n\n"

        "## Useful For\n"
        "- Playing back previously recorded signal data.\n"
        "- Testing signal processing chains with known input data.\n"
        "- Offline analysis of captured signals.\n\n"

        "## Examples\n"
        "- Read CF32 IQ recording:\n"
        "  Config: File Path='recording.raw', Data Type=CF32, Batch Size=8192, Loop=true\n"
        "  Output: CF32[8192]\n"
        "- Read CI16 IQ recording:\n"
        "  Config: File Path='capture.raw', Data Type=CI16, Batch Size=4096, Loop=false\n"
        "  Output: CI16[4096]\n\n"

        "## Implementation\n"
        "FileReader Module -> Output Buffer\n"
        "1. Opens the specified file in binary read mode.\n"
        "2. Reads Batch Size samples of the specified data type per cycle.\n"
        "3. When end of file is reached, either loops back or yields."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_FILE_READER_BLOCK_HH
